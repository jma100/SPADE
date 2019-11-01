"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import torch
import numpy as np
import random
import json

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        if opt.use_acgan:
            self.mapping = {8: 0, 9: 1, 11: 2, 15: 3, 16: 4, 18: 5, 19: 6, 20: 7, 23: 8, 24: 9, 25: 10, 28: 11, 29: 12, 31: 13, 32: 14, 34: 15, 36: 16, 37: 17, 38: 18, 39: 19, 40: 20, 42: 21, 43: 22, 44: 23, 45: 24, 46: 25, 48: 26, 50: 27, 51: 28, 54: 29, 56: 30, 57: 31, 58: 32, 59: 33, 60: 34, 63: 35, 64: 36, 65: 37, 66: 38, 67: 39, 68: 40, 70: 41, 71: 42, 72: 43, 74: 44, 75: 45, 76: 46, 82: 47, 83: 48, 86: 49, 87: 50, 90: 51, 93: 52, 96: 53, 98: 54, 99: 55, 100: 56, 101: 57, 107: 58, 108: 59, 109: 60, 111: 61, 113: 62, 116: 63, 118: 64, 119: 65, 120: 66, 121: 67, 122: 68, 125: 69, 126: 70, 130: 71, 132: 72, 133: 73, 134: 74, 135: 75, 136: 76, 138: 77, 139: 78, 140: 79, 143: 80, 144: 81, 145: 82, 146: 83, 147: 84, 148: 85, 149: 86, 150: 87}
        paths_dict = self.get_paths(opt)
        label_paths = paths_dict['label_paths']
        image_paths = paths_dict['image_paths']
        instance_paths = paths_dict['instance_paths']
        if opt.use_scene:
            scene_paths = paths_dict['scene_paths']
        if opt.use_depth:
            depth_paths = paths_dict['depth_paths']

#        util.natural_sort(label_paths)
#        util.natural_sort(image_paths)
#        if not opt.no_instance:
#            util.natural_sort(instance_paths)

        if opt.use_scene:
            self.scene_mapping = {'bathroom':0, 'bedroom':1, 'kitchen':2, 'living_room':3, 'childs_room':4, 'dining_room':5, 'dorm_room':6, 'hotel_room':7}
        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        if opt.use_scene:
            scene_paths = scene_paths[:opt.max_dataset_size]
            self.scene_paths = scene_paths
        if opt.use_depth:
            #util.natural_sort(depth_paths)
            depth_paths = depth_paths[:opt.max_dataset_size]
            self.depth_paths = depth_paths

        if opt.use_material:
            util.natural_sort(material_paths)
            material_paths = material_paths[:opt.max_dataset_size]
            self.material_paths = material_paths

        if opt.use_illumination:
            util.natural_sort(illumination_paths)
            illumination_paths = illumination_paths[:opt.max_dataset_size]
            self.illumination_paths = illumination_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext= os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def load_mapping(self):
        with open(self.opt.mapping_path, 'r') as f:
            data = json.load(f)
        return data

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # object class label
        if self.opt.use_acgan:
            object_class = self.mapping[int(label_path.split('/')[-2])]
            object_tensor = torch.FloatTensor(1).fill_(object_class)
            object_tensor = object_tensor.expand_as(label_tensor)

        # object class label
        if self.opt.use_scene:
            scene_class = self.scene_mapping[self.scene_paths[index]]
            scene_tensor = torch.FloatTensor(1).fill_(scene_class)
            scene_tensor = scene_tensor.expand_as(label_tensor)

        # input image (real images)
        image_path = self.image_paths[index]
        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        # material
        if self.opt.use_material:
            material_path = self.material_paths[index]
            material = Image.open(material_path)
            material_tensor = transform_label(material) * 255.0

        # depth (processing)
        if self.opt.use_depth:
            depth_path = self.depth_paths[index]
            depth = Image.open(depth_path)
            segm_data = np.array(label)
            if 3 in np.unique(segm_data): # 3 is the label for sky
                # Set sky depth value to minimum depth value in non-sky area
                im_mode = depth.mode
                depth_data = np.array(depth)
                # first set sky depth to max to find out the minimum of non-sky area
                sky_set_to_max = depth_data
                sky_set_to_max[segm_data == 3] = 65535
                min_val = np.min(sky_set_to_max)
                # set sky depth to the minimum
                depth_processed = depth_data
                depth_processed[segm_data == 3] = min_val
                depth = Image.fromarray(depth_processed, mode=im_mode)
            
            depth_tensor = transform_label(depth).float() * 255.0
            depth_tensor = ((depth_tensor-torch.min(depth_tensor))/(torch.max(depth_tensor)-torch.min(depth_tensor))-0.5) *2.0

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        if self.opt.real_background:
            # foreground, feed into encoder
            fg_tensor = image_tensor * label_tensor.long().float()
            # background, combined with generated foreground
            bg_tensor = image_tensor * (1 - label_tensor.long()).float()

        if self.opt.no_background:
            # foreground, feed into encoder
            fg_tensor = image_tensor * label_tensor.long().float()            

        if self.opt.add_hint:
            hint_tensor = image_tensor.clone()
            left, right = self.opt.crop_size // 2 - 15, self.opt.crop_size // 2 + 15
            up, down = left, right
            hint_tensor[ :, :up, :] = 0
            hint_tensor[ :, down:, :] = 0
            hint_tensor[ :, :, :left] = 0
            hint_tensor[ :, :, right:] = 0

            if self.opt.random_hint:
                random_hint_tensor = image_tensor.clone()
                random_hint_tensor[:, :, :] = 0
                rand_u, rand_l = random.randint(0, self.opt.crop_size-31), random.randint(0, self.opt.crop_size-31)
                rand_d, rand_r = rand_u+down-up, rand_l + right -left
                random_hint_tensor[:, rand_u:rand_d, rand_l:rand_r] = hint_tensor[:, up:down, left:right]



        # illumination
        if self.opt.use_illumination:
            illumination_path = self.illumination_paths[index]
            illumination = Image.open(illumination_path).convert('L')
            transform_illu = get_transform(self.opt, params, num_channel=1)
            illumination_tensor = transform_illu(illumination)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)
        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        if self.opt.use_depth:
            input_dict['depth'] = depth_tensor
        if self.opt.use_material:
            input_dict['material'] = material_tensor
        if self.opt.use_illumination:
            input_dict['illumination'] = illumination_tensor
        if self.opt.add_hint:
            input_dict['hint'] = hint_tensor
        if self.opt.random_hint:
            input_dict['hint'] = random_hint_tensor
        if self.opt.real_background:
            input_dict['fg'] = fg_tensor
            input_dict['bg'] = bg_tensor
        if self.opt.use_acgan:
            input_dict['object'] = object_tensor
            input_dict['object_class'] = object_class
        if self.opt.use_scene:
            input_dict['scene'] = scene_tensor
        if self.opt.no_background:
            input_dict['image'] = fg_tensor

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
