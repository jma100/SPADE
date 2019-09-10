"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import os

class ADE20KObjDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(load_size=128)
        parser.set_defaults(crop_size=128)
        parser.set_defaults(display_winsize=128)
        parser.set_defaults(label_nc=1)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(nThreads=16)
        parser.set_defaults(margin=16)
        parser.set_defaults(use_vae=True)
        parser.set_defaults(no_acgan_loss=True)
        parser.add_argument('--train_list', type=str, help='import list of training folders')
        parser.add_argument('--object_info', type=str, help='object name: object ade id, object min training size')
        parser.add_argument('--obj_load_size', type=int, default=128, help='load size for cropped objects')
        parser.add_argument('--obj_crop_size', type=int, default=128, help='crop size for loaded objects')
        parser.add_argument('--max_object_per_image', type=int, default=1, help='number of objects per image during training')
        parser.set_defaults(is_object=True)
        parser.add_argument('--metadata_list', type=str, help='import list of metadata files (depth, normal, material, part)')
        return parser

    def get_paths(self, opt):
        with open(opt.train_list,'r') as f:
            training_list = f.read().split('\n')
        if training_list[-1]=='':
            training_list = training_list[:-1]
        image_paths = []
        label_paths = []
        for i,p in enumerate(training_list):
            if i % 2 == 0:
                image_paths.append(p)
            else:
                label_paths.append(p)


        instance_paths = [] # don't use instance map for ade20k
        paths = {'label': label_paths, 'image': image_paths, 'instance': instance_paths}

        if opt.metadata_list != None:
            with open(opt.metadata_list,'r') as f:
                metadata_list = f.read().split('\n')
            if metadata_list[-1]=='':
                metadata_list = metadata_list[:-1]
            depth_paths = []
            normal_paths = []
            material_paths = []
            part_paths = []
            for i,p in enumerate(metadata_list):
                if 'depth' in p and opt.use_depth:
                    depth_paths.append(p)
                elif 'normal' in p and opt.use_normal:
                    normal_paths.append(p)
                elif 'material' in p and opt.use_material:
                    material_paths.append(p)
                elif 'part' in p and opt.use_part:
                    part_paths.append(p)

            paths['depth'] = depth_paths
            paths['normal'] = normal_paths
            paths['material'] = material_paths
            paths['part'] = part_paths

        return paths

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc
        if self.opt.use_material:
            material = input_dict['material']
            material = material - 1
            material[material == -1] = self.opt.material_nc

