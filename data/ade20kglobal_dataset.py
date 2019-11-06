"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class ADE20KGlobalDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
#        if is_train:
#            parser.set_defaults(load_size=286)
#        else:
#            parser.set_defaults(load_size=256)
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=150)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(nThreads=16)
        parser.set_defaults(margin=16)
        parser.set_defaults(use_vae=True)
        parser.add_argument('--train_list', type=str, help='import list of training folders')
        parser.add_argument('--obj_load_size', type=int, default=128, help='load size for cropped objects')
        parser.add_argument('--obj_crop_size', type=int, default=128, help='crop size for loaded objects')
        parser.add_argument('--max_object_per_image', type=int, default=1, help='number of objects per image during training')
        parser.add_argument('--is_object', action='store_true', help='indicate if the pix2pix model is object renderer or global renderer')
        parser.add_argument('--pretrain_object', type=str, help='folder of pretrained object model checkpoints')
        parser.add_argument('--pretrain_stuff', type=str, help='folder of pretrained stuff model checkpoints')
        parser.add_argument('--use_acgan_loss', action='store_true', help='add acgan loss or not')
        parser.add_argument('--use_scene', action='store_true', help='input scene category or not')
        parser.add_argument('--use_instance_crop', action='store_true', help='use instance segmentation in cropping or not')
        parser.add_argument('--instance_dir', type=str, help='path to instance folder for cropping')
        parser.add_argument('--instance_conversion', type=str, help='path to 150 class to 100 class conversion')
        parser.add_argument('--use_object_z', action='store_true', help='set to True when loading an object generator trained with stylegan')
        parser.add_argument('--use_stuff_z', action='store_true', help='set to True when loading an stuff generator trained with stylegan')
        

        parser.set_defaults(use_acgan=True)
        parser.set_defaults(use_scene=True)
        parser.set_defaults(acgan_nc=88)
        parser.add_argument('--scene_nc', type=int, help='number of scene classes')
        parser.set_defaults(scene_nc=8)
        return parser

    def get_paths(self, opt):
        with open(opt.train_list,'r') as f:
            training_list = f.read().split('\n')[:-1]
        image_paths = []
        label_paths = []
        for i,p in enumerate(training_list):
            if 'semantic' in p or 'png' in p:
                label_paths.append(p)
            else:
                image_paths.append(p)

        instance_paths = [] # don't use instance map for ade20k
        paths = {'label': label_paths, 'image': image_paths, 'instance': instance_paths}
        return paths

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc
