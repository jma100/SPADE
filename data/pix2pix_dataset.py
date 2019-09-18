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
            self.mapping = self.load_mapping()
        if opt.is_object:
            self.object_info = self.load_mapping(opt.object_info)

        paths = self.get_paths(opt)
        label_paths, image_paths, instance_paths = paths['label'], paths['image'], paths['instance']        


        if self.opt.dataroot != './datasets/cityscapes/':
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
            depth_paths = paths['depth']
            depth_paths = depth_paths[:opt.max_dataset_size]
            self.depth_paths = depth_paths

        if opt.use_normal:
            normal_paths = paths['normal']
            normal_paths = normal_paths[:opt.max_dataset_size]
            self.normal_paths = normal_paths

        if opt.use_material:
            material_paths = paths['material']
            material_paths = material_paths[:opt.max_dataset_size]
            self.material_paths = material_paths

        if opt.use_part:
            part_paths = paths['part']
            part_paths = part_paths[:opt.max_dataset_size]
            self.part_paths = part_paths

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
        label_data = np.array(label)

        # input image (real images)
        image_path = self.image_paths[index]
        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        if self.opt.use_image != '':
            encode = Image.open(self.opt.use_image)
            encode = encode.convert('RGB')
        # depth (processing)
        if self.opt.use_depth:
            depth_path = self.depth_paths[index]
            depth = Image.open(depth_path)
        if self.opt.use_part:
            part_path = self.part_paths[index]
            part = Image.open(part_path)
        if self.opt.use_normal:
            normal_path = self.normal_paths[index]
            normal = Image.open(normal_path)
        if self.opt.use_material:
            material_path = self.material_paths[index]
            material = Image.open(material_path)

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

        # objects
        if self.opt.is_object:
            objects_exist = np.unique(label_data)
            margin = self.opt.margin
            all_objects = dict()
            for obj, (obj_id, min_size) in self.object_info.items():
                if obj_id in objects_exist:
                    # find instance map
                    blobs = label_data == obj_id
                    instance_data, nlabels = ndimage.label(blobs)

                    height, width = label_data.shape

                    for instance_id in range(1, nlabels+1):
                        up, down, left, right = self.bbox(instance_data, instance_id)
                        up, down, left, right = max(0, up-margin), min(height, down+margin), max(0, left -margin), min(width, right + margin)
                        if down-up < min_size or right-left < min_size:
                            continue
                        data_instance = dict()
                        instance_name = obj + '_%03d' % instance_id
                        cropped = image.crop((left, up, right, down))
                        mask_out = (instance_data == instance_id).astype(int)
                        cropped_mask = mask_out[up:down, left:right]
                        cropped_label = Image.fromarray(cropped_mask.astype('uint8'))
                        obj_params = get_params(self.opt, (right-left, down-up))
                        transform_obj_image = get_transform(self.opt, obj_params)
                        transform_obj_label = get_transform(self.opt, obj_params, method=Image.NEAREST, normalize=False)
                        new_left = int(float(left)/width*self.opt.crop_size)
                        new_right = int(float(right)/width*self.opt.crop_size)
                        new_up = int(float(up)/height*self.opt.crop_size)
                        new_down = int(float(down)/height*self.opt.crop_size)
                        data_instance['bbox'] = [new_left, new_up, new_right, new_down]
                        data_instance['image'] = transform_obj_image(cropped)
                        data_instance['label'] = transform_obj_label(cropped_label) * 255.0
                        if self.opt.use_acgan:
                            object_class = self.mapping[obj]
                            object_tensor = torch.FloatTensor(1).fill_(object_class)
                            object_tensor = object_tensor.expand_as(data_instance['label'])
                            data_instance['object'] = object_tensor
                            data_instance['object_class'] = object_class
                        if self.opt.no_background:
                            data_instance['image'] = data_instance['image'] * data_instance['label'].long().float()
                        if self.opt.real_background or self.opt.encode_background:
                            data_instance['fg'] = data_instance['image'] * data_instance['label'].long().float()
                            data_instance['bg'] = data_instance['image'] * (1-data_instance['label'].long()).float()
                            
                        data_instance['instance'] = 0
                        data_instance['path'] = image_path
                        if self.opt.use_depth:
                            cropped_depth = depth.crop((left, up, right, down))
                            data_instance['depth'] = transform_obj_label(cropped_depth).float()
                        if self.opt.use_part:
                            cropped_part = part.crop((left, up, right, down))
                            data_instance['part'] = transform_obj_label(cropped_part).float()
                        if self.opt.use_normal:
                            cropped_normal = normal.crop((left, up, right, down))
                            data_instance['normal'] = transform_obj_label(cropped_normal).float()
                        if self.opt.use_material:
                            cropped_material = material.crop((left, up, right, down))
                            data_instance['material'] = transform_obj_label(cropped_material).float() 
                        if self.opt.position_encode:
                            _, h, w = data_instance['label'].size()
                            data_instance['pos_x'] = torch.LongTensor(np.array([[[j for j in range(w)] for i in range(h)]]))
                            data_instance['pos_y'] = torch.LongTensor(np.array([[[i for j in range(w)] for i in range(h)]]))
                        if self.opt.use_image != "":
                            encode_tensor = transform_obj_image(encode)
                            data_instance['encode'] = encode_tensor
                        all_objects[instance_name] = data_instance

            if self.opt.phase == 'train':
                chosen = np.random.choice(list(all_objects.keys()), self.opt.max_object_per_image)
                input_dict = all_objects[chosen[0]]
            else:
                input_dict = all_objects['bed_001']
           # implement
#            for i in range(self.opt.max_object_per_image):
#                all_objects[chosen[i]]['object_name'] = chosen[i]
#                input_dict['object_%03d' % i] = all_objects[chosen[i]]
            return input_dict
        # Label Image
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # object class label
        if self.opt.use_acgan:
            object_class = self.mapping[label_path.split('/')[-3]]
            object_tensor = torch.FloatTensor(1).fill_(object_class)
            object_tensor = object_tensor.expand_as(label_tensor)

        # input image (real images)
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        if self.opt.real_background or self.opt.encode_background:
            # foreground, feed into encoder
            fg_tensor = image_tensor * label_tensor.long().float()
            # background, combined with generated foreground
            bg_tensor = image_tensor * (1 - label_tensor.long()).float()

        if self.opt.no_background:
            # foreground, feed into encoder
            fg_tensor = image_tensor * label_tensor.long().float()  

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }
        if self.opt.use_depth:
            depth_path = self.depth_paths[index]
            depth = Image.open(depth_path)
            depth_tensor = transform_label(depth).float()
            input_dict['depth'] = depth_tensor
        if self.opt.use_normal:
            normal_path = self.normal_paths[index]
            normal = Image.open(normal_path)
            normal_tensor = transform_label(normal).float()
            input_dict['normal'] = normal_tensor
        if self.opt.use_material:
            material_path = self.material_paths[index]
            material = Image.open(material_path)
            material_tensor = transform_label(material).float()
            input_dict['material'] = material_tensor
        if self.opt.use_part:
            part_path = self.part_paths[index]
            part = Image.open(part_path)
            part_tensor = transform_label(part).float()
            input_dict['part'] = part_tensor
        if self.opt.use_acgan:
            input_dict['object'] = object_tensor
            input_dict['object_class'] = object_class
        if self.opt.real_background or self.opt.encode_background:
            input_dict['fg'] = fg_tensor
            input_dict['bg'] = bg_tensor
        if self.opt.no_background:
            input_dict['image'] = fg_tensor
        if self.opt.position_encode:
            _, h, w = label_tensor.size()
            input_dict['pos_x'] = torch.LongTensor(np.array([[[j for j in range(w)] for i in range(h)]]))
            input_dict['pos_y'] = torch.LongTensor(np.array([[[i for j in range(w)] for i in range(h)]]))
        if self.opt.use_image != "":
            encode_tensor = transform_image(encode)
            input_dict['encode'] = encode_tensor  
        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
