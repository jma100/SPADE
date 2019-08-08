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
import json
from scipy import ndimage

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        if opt.use_acgan:
            self.mapping = self.load_mapping(opt.mapping_path)
        if opt.dataset_mode == 'ade20kglobal':
            self.object_info = self.load_mapping(opt.object_info)
        paths = self.get_paths(opt)
        label_paths, image_paths, instance_paths = paths['label'], paths['image'], paths['instance']        

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
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def load_mapping(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def bbox(self, img, instance_id):
        a = np.where(img == instance_id)
        bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
        return bbox

    def __getitem__(self, index):
        input_dict = dict()        

        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')
        if opt.dataset_mode == 'ade20kglobal':
            label = label.resize((self.opt.crop_size, self.opt.crop_size))
        label_data = np.array(label)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        if self.opt.use_acgan:
            object_class = self.mapping[label_path.split('/')[-3]]
            object_tensor = torch.FloatTensor(1).fill_(object_class)
            object_tensor = object_tensor.expand_as(label_tensor)

        # input image (real images)
        image_path = self.image_paths[index]
        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')
        if self.opt.dataset_mode == 'ade20kglobal':
            image = image.resize((self.opt.crop_size, self.opt.crop_size))

        # objects
        if self.opt.dataset_mode == 'ade20kglobal':
            objects_exist = np.unique(label_data)
            flip = params['flip']
            for obj, (obj_id, obj_threshold) in self.object_id.iteritems():
                if obj_id in objects_exist:
                    data_obj = dict()
                    # find instance map
                    blobs = label_data == obj_id
                    instance_data, nlabels = ndimage.label(blobs)

                    height, width = segm_data.shape
                    min_size = obj_threshold[obj]

                    for instance_id in range(1, nlabels+1):
                        up, down, left, right = self.bbox(instance_data, instance_id)
                        if down-up+self.opt.margin*2 < min_size or right-left+self.opt.margin*2 < min_size:
                            continue
                        data_instance = dict()
                        instance_name = obj + '_%03d' % instance_id
                        up, down, left, right = max(0, up-margin), min(height, down+margin), max(0, left -margin), min(width, right + margin)
                        cropped = img.crop((left, up, right, down))
                        mask_out = (instance_data == instance_id).astype(int)
                        cropped_mask = mask_out[up:down, left:right]
                        cropped_label = Image.fromarray(cropped_mask.astype('uint8'))
                        obj_params = get_params(self.opt, (right-left, down-up), use_object=True, use_flip=flip)
                        transform_obj_image = get_transform(self.opt, params)
                        transform_obj_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                        data_instance['bbox'] = [left, up, right, down]
                        data_instance['image'] = transform_obj_image(cropped)
                        data_instance['label'] = transform_obj_label(cropped_label) * 255.0
                        data_obj[instance_name] = data_instance

                    input_dict[obj] = data_obj

        # depth (processing)
        if self.opt.use_depth:
            depth_path = self.depth_paths[index]
            depth = Image.open(depth_path)
#            if depth.mode=='I':
#                data = np.array(depth).astype(float)/10
#                data = data.astype(np.uint8)
#                depth = Image.fromarray(data).convert('L')
            depth_tensor = transform_label(depth).float()
            depth_tensor[depth_tensor == 0] = torch.max(depth_tensor)
            depth_tensor = ((depth_tensor-torch.min(depth_tensor))/torch.max(depth_tensor)-0.5) *2.0

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

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

        if self.opt.real_background:
            # foreground, feed into encoder
            fg_tensor = image_tensor * label_tensor.long().float()
            # background, combined with generated foreground
            bg_tensor = image_tensor * (1 - label_tensor.long()).float()

        if self.opt.no_background:
            # foreground, feed into encoder
            fg_tensor = image_tensor * label_tensor.long().float()  

        data_global = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }
        if self.opt.use_depth:
            data_global['depth'] = depth_tensor
        if self.opt.use_acgan:
            data_global['object'] = object_tensor
            data_global['object_class'] = object_class
        if self.opt.real_background:
            data_global['fg'] = fg_tensor
            data_global['bg'] = bg_tensor
        if self.opt.no_background:
            data_global['image'] = fg_tensor
        # Give subclasses a chance to modify the final output
        self.postprocess(data_global)
        input_dict['global'] = data_global
        if self.opt.dataset_mode == 'ade20kglobal':
            return input_dict
        else:
            return data_global
    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
