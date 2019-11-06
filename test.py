"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from models.merge_model import MergeModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

#model = Pix2PixModel(opt)
model = MergeModel(opt)
for name, param in model.named_parameters():
    if name == 'netA.conv_merge.weight':
        weights = param
import pdb; pdb.set_trace()
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    for obj, obj_data in data_i.items():
        opt.is_object = True if obj != 'global' else False
        data_i[obj] = model.preprocess_input(obj_data)

    # Inference each object
    opt.is_object = True
    for n in range(opt.max_object_per_image):
        name = 'object_%03d' % n
        data_i[name]['generated'] =  model.net_object(data_i[name], mode='inference')

    # Inference stuff
    opt.is_object = False
    data_i['global']['generated'] = model.net_global(data_i['global'], mode='inference')

    generated = model(data_i, mode='inference')

    img_path = data_i['global']['path']
    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['global']['label'][b]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
