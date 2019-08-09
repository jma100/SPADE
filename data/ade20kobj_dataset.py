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
        parser.add_argument('--depth_dir', type=str,
                            help='path to the directory that contains depth images')
        parser.add_argument('--more_data', type=str,
                            help='additional directory with more data, usually interiornet object')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        all_images = make_dataset(root, recursive=True, read_cache=False, write_cache=False)
        image_paths = []
        label_paths = []
        depth_paths = []
        material_paths = []
        illumination_paths = []
        for p in all_images:
#            if '_%s_' % phase not in p:
#                continue
            if p.endswith('.jpg'):
                image_paths.append(p)
            elif p.endswith('.png'):
                label_paths.append(p)
        instance_paths = []  # don't use instance map for ade20k

        if opt.more_data != None:
            more_images = make_dataset(opt.more_data, recursive=True, read_cache=False, write_cache=False)
            for p in more_images:
                if p.endswith('.jpg'):
                    image_paths.append(p)
                elif p.endswith('.png'):
                    label_paths.append(p)

        if opt.use_material:
            material_root = '/data/vision/torralba/scratch2/jingweim/unifiedparsing/data/ade'
            material_paths = [os.path.join(material_root, p.split('/')[-1][:-4], 'material_result.png') for p in image_paths]

        if opt.use_depth:
            depth_paths = make_dataset(opt.depth_dir, recursive=False, read_cache=True)

        paths = {'label': label_paths, 'image': image_paths, 'depth': depth_paths, 'material': material_paths, 'instance': instance_paths, 'illumination': illumination_paths}

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

