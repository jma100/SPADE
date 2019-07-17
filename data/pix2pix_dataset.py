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

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths, depth_paths, material_paths, illumination_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

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

        if opt.use_depth:
            util.natural_sort(depth_paths)
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

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

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
            im_mode = depth.mode
            if self.opt.mask_sky:
#            if self.opt.dataset_mode == 'ade20k':
                # set sky depth to min
                depth_data = np.array(depth)
                data_type = depth_data.dtype
                segm_data = np.array(label).astype(data_type)
                mask = 1-(segm_data==3)
                depth_data = depth_data * mask
                min_val = np.partition(np.unique(depth_data), 2)[1]
                depth_data[depth_data == 0] = min_val
                depth = Image.fromarray(depth_data.astype(data_type), mode=im_mode)
#            if depth.mode=='I':
#                data = np.array(depth).astype(float)/10
#                data = data.astype(np.uint8)
#                depth = Image.fromarray(data).convert('L')
            depth_tensor = transform_label(depth).float() * 255.0
            if self.opt.dataset_mode == 'interiornet':
                depth_tensor[depth_tensor == 0] = torch.max(depth_tensor)
            depth_tensor = ((depth_tensor-torch.min(depth_tensor))/torch.max(depth_tensor)-0.5) *2.0

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

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

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
