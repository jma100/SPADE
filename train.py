"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.merge_trainer import MergeTrainer
import torch
from torchsummary import summary

torch.multiprocessing.set_sharing_strategy('file_system')

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))


# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
merge_trainer = MergeTrainer(opt) # implement

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)


for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Process background model data
        opt.is_object = False
        data_i['global'] = merge_trainer.merge_model.module.preprocess_input(data_i['global'])

        # Process object model data
        opt.is_object = True
#        data_i['object'] = merge_trainer.merge_model.module.preprocess_object_input(obj_data)

        for obj, obj_data in data_i['object'].items():
#            opt.is_object = True if obj != 'global' else False
            data_i['object'][obj] = merge_trainer.merge_model.module.preprocess_input(obj_data)
        z = []
        # Training
        # train each object
        opt.is_object = True
        for n in range(opt.max_object_per_image):
            name = 'object_%03d' % n
            # train object generator
            if i % opt.D_steps_per_G == 0:
                merge_trainer.run_object_generator_one_step(data_i['object'][name])
                data_i['object'][name]['generated'] = merge_trainer.object_generated
                data_i['object'][name]['features'] = merge_trainer.object_features
                data_i['object'][name]['z'] = merge_trainer.object_z
                if len(z) == 0:
                    z = data_i['object'][name]['z']
                else:
                    z = torch.cat([z, data_i['object'][name]['z']], 1)
            # train object discriminator
            if not opt.load_pretrain:
                merge_trainer.run_object_discriminator_one_step(data_i['object'][name])
        torch.cuda.empty_cache()

        opt.is_object = False
        # train global generator
        if i % opt.D_steps_per_G == 0:
            merge_trainer.run_global_generator_one_step(data_i['global'])
            data_i['global']['generated'] = merge_trainer.global_generated
            data_i['global']['features'] = merge_trainer.global_features
            if opt.use_stuff_vae:
                data_i['global']['z'] = merge_trainer.global_z
                z = torch.cat([z, data_i['global']['z']], 1)
            data_i['z'] = z

        # train global discriminator
        if not opt.load_pretrain:
            merge_trainer.run_global_discriminator_one_step(data_i['global'])
        torch.cuda.empty_cache()

        # train merge step
        if i % opt.D_steps_per_G == 0:
            merge_trainer.run_overall_one_step(data_i)
            data_i['generated'] = merge_trainer.generated

        # train overall discriminator
        merge_trainer.run_discriminator_one_step(data_i)

        torch.cuda.empty_cache()
        # Visualizations
        if iter_counter.needs_printing():
            if opt.load_pretrain:
                losses = merge_trainer.get_latest_losses() 
            else:
                losses = merge_trainer.get_latest_losses()[0]
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['global']['label'])])
            visuals['object'] = dict()
            for n in range(opt.max_object_per_image):
                name = 'object_%03d' % n
                for k, padded in enumerate(data_i['object'][name]['padded']):
                    if padded.item() == 1:
                        data_i['object'][name]['generated'][k, :, :, :] = -1
                visuals['object'][name] = data_i['object'][name]['generated']
            visuals['synthesized_global'] = data_i['global']['generated']
            visuals['combined_image'] = data_i['generated']
            visuals['real_image'] = data_i['global']['image']

            visualizer.display_results_one_by_one(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            merge_trainer.save('latest')
            iter_counter.record_current_iter()

    merge_trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        merge_trainer.save('latest')
        merge_trainer.save(epoch)

print('Training was successfully finished.')
