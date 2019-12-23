"""
                    iown = min(size, down-size)
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
            self.mapping = {8: 0, 9: 1, 11: 2, 15: 3, 16: 4, 18: 5, 19: 6, 20: 7, 23: 8, 24: 9, 25: 10, 28: 11, 29: 12, 31: 13, 32: 14, 34: 15, 36: 16, 37: 17, 38: 18, 39: 19, 40: 20, 42: 21, 43: 22, 44: 23, 45: 24, 46: 25, 48: 26, 50: 27, 51: 28, 54: 29, 56: 30, 57: 31, 58: 32, 59: 33, 60: 34, 63: 35, 64: 36, 65: 37, 66: 38, 67: 39, 68: 40, 70: 41, 71: 42, 72: 43, 74: 44, 75: 45, 76: 46, 82: 47, 83: 48, 86: 49, 87: 50, 90: 51, 93: 52, 96: 53, 98: 54, 99: 55, 100: 56, 101: 57, 107: 58, 108: 59, 109: 60, 111: 61, 113: 62, 116: 63, 118: 64, 119: 65, 120: 66, 121: 67, 122: 68, 125: 69, 126: 70, 130: 71, 132: 72, 133: 73, 134: 74, 135: 75, 136: 76, 138: 77, 139: 78, 140: 79, 143: 80, 144: 81, 145: 82, 146: 83, 147: 84, 148: 85, 149: 86, 150: 87}
            self.yes_classes = [8, 9, 11, 15, 16, 19, 20, 23, 24, 25, 28, 31, 32, 34, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 50, 51, 54, 56, 57, 58, 59, 63, 65, 66, 67, 68, 70, 71, 72, 74, 75, 76, 82, 83, 86, 87, 90, 93, 96, 98, 99, 108, 109, 111, 113, 116, 119, 120, 121, 122, 125, 126, 130, 133, 134, 135, 136, 138, 139, 140, 143, 144, 145, 147, 148, 149, 150]
        if opt.use_instance_crop:
            mapping = open(self.opt.instance_conversion, 'r').read().split('\n')[1:-1]
            self.object_to_instance = {int(f.split()[1]): int(f.split()[0]) for f in mapping}
        paths = self.get_paths(opt)
        label_paths, image_paths, instance_paths = paths['label'], paths['image'], paths['instance']        

#        util.natural_sort(label_paths)
#        util.natural_sort(image_paths)
#        if not opt.no_instance:
#            util.natural_sort(instance_paths)

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
#            util.natural_sort(depth_paths)
            depth_paths = depth_paths[:opt.max_dataset_size]
            self.depth_paths = depth_paths

        if opt.use_scene:
            scene_file = '/data/vision/torralba/virtualhome/realvirtualhome/SPADE_old/datasets/ADE150Indoor/sceneCategories.txt'
            with open(scene_file, 'r') as f:
                lines = f.read().split('\n')[:-1]
            self.scene_mapping = {line.split()[0]:line.split()[1] for line in lines}
            self.scene_name_to_index = {'bathroom':0, 'bedroom':1, 'kitchen':2, 'living_room':3, 'childs_room':4, 'dining_room':5, 'dorm_room':6, 'hotel_room':7}
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
        params = get_params(self.opt, label.size)
        if self.opt.no_flip:
            params['flip'] = False
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
        assert label.size == image.size

        # Depth
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

            depth_data = np.array(depth)            
            depth_tensor = transform_label(depth).float() * 255.0
            depth_tensor = ((depth_tensor-torch.min(depth_tensor))/(torch.max(depth_tensor)-torch.min(depth_tensor))-0.5) *2.0

        # objects
        if self.opt.dataset_mode == 'ade20kglobal' and self.opt.use_instance_crop:
            objects_exist = list(set(np.unique(label_data)).intersection(set(self.yes_classes)))
            # get scene class
            if self.opt.use_scene:
                name = label_path.split('/')[-1][:-4]
                scene_name = self.scene_mapping[name]
                scene_class = self.scene_name_to_index[scene_name]
            flip = params['flip']
            margin = self.opt.margin
            all_objects = dict()
            height, width = label_data.shape
            instance_path = os.path.join(self.opt.instance_dir, label_path.split('/')[-1])
            instance = Image.open(instance_path)
            instance_data = np.array(instance)
            instance_rgbs = set( tuple(v) for m2d in instance_data for v in m2d )
            img_data = np.array(image)
            all_instances = []
            for obj_id in objects_exist:
                # randomly choose instance
                instance_ids = np.unique([s[1] for s in instance_rgbs if s[0]==self.object_to_instance[obj_id]])
                for instance_id in instance_ids:
                    instance_name = '%03d_%03d' % (obj_id, instance_id)
                    all_instances.append(instance_name)
            np.random.shuffle(all_instances)

            for chosen_object in all_instances:
                if len(all_objects.keys()) >= self.opt.max_object_per_image:
                    continue
                obj_id, instance_id = chosen_object.split('_')
                obj_id, instance_id = int(obj_id), int(instance_id)
                mask = (instance_data[:, :, 0] == self.object_to_instance[obj_id]).astype(int)
                mask_rgb = np.zeros(instance_data.shape)
                mask_rgb[...,0] = mask
                mask_rgb[...,1] = mask
                mask_rgb[...,2] = mask
                instance_masked = (instance_data * mask_rgb)[..., 1]
                up, down, left, right = self.bbox(instance_masked, instance_id)
                w_padded, h_padded = False, False
                if down-up < 20 or right-left < 20:
                    continue
                data_instance = dict()
                # Add margins
                up, down, left, right = max(0, up-margin), min(height, down+margin), max(0, left -margin), min(width, right + margin)
                w, h = right-left, down-up
                if w > h:
                    # Make square
                    center = (down+up)//2
                    up = center - w//2
                    down = center + (w-w//2)
                    assert right-left == down-up
                    if up < 0 or down > height:
                        img_data_pad = np.pad(img_data, ((height, height), (0,0), (0,0)), 'reflect')
                        img_new = Image.fromarray(img_data_pad)
                        up += height
                        down += height
                        instance_masked_new = np.pad(instance_masked, ((height, height), (0,0)), 'constant')
                        if self.opt.use_depth:
                            depth_data_pad = np.pad(depth_data, ((height, height), (0,0)), 'reflect')
                            depth_new = Image.fromarray(depth_data_pad)
                        h_padded = True
                    else:
                        img_new = image
                        instance_masked_new = instance_masked
                        if self.opt.use_depth:
                            depth_new = depth
	
                elif h > w:
                    # Make square
                    center = (left+right)//2
                    left = center - h//2
                    right = center + (h-h//2)
                    assert right-left == down-up
                    if left < 0 or right > width:
                        img_data_pad = np.pad(img_data, ((0,0), (width, width), (0,0)), 'reflect')
                        img_new = Image.fromarray(img_data_pad)
                        left += width
                        right += width
                        instance_masked_new = np.pad(instance_masked, ((0,0), (width, width)), 'constant')
                        if self.opt.use_depth:
                            depth_data_pad = np.pad(depth_data, ((0, 0), (width, width)), 'reflect')
                            depth_new = Image.fromarray(depth_data_pad)
                        w_padded = True
                    else:
                        img_new = image
                        instance_masked_new = instance_masked
                        if self.opt.use_depth:
                            depth_new = depth
                else:
                    img_new = image
                    instance_masked_new = instance_masked
                    if self.opt.use_depth:
                        depth_new = depth

                assert down-up == right-left
                cropped = img_new.crop((left, up, right, down))
                mask_out = (instance_masked_new == instance_id).astype(int)
                cropped_mask = mask_out[up:down, left:right]
                cropped_label = Image.fromarray(cropped_mask.astype('uint8'))
                if self.opt.use_depth:
                    cropped_depth = depth_new.crop((left, up, right, down))
                obj_params = get_params(self.opt, (right-left, down-up), use_object=True, use_flip=flip)
                transform_obj_image = get_transform(self.opt, obj_params, use_object=True)
                transform_obj_label = get_transform(self.opt, obj_params, method=Image.NEAREST, normalize=False, use_object=True)
#                if not w_padded:
#                    new_left = int(float(left)/width*self.opt.crop_size)
#                    new_right = int(float(right)/width*self.opt.crop_size)
#                else:
#                    new_left = int(float(left-width)/width*self.opt.crop_size)
#                    new_right = int(float(right-width)/width*self.opt.crop_size)
#                if not h_padded:
#                    new_up = int(float(up)/height*self.opt.crop_size)
#                    new_down = int(float(down)/height*self.opt.crop_size)
#                else:
#                    new_up = int(float(up-height)/height*self.opt.crop_size)
#                    new_down = int(float(down-height)/height*self.opt.crop_size)
                new_left = float(left)/width*self.opt.crop_size
                new_right = float(right)/width*self.opt.crop_size
                new_up = float(up)/height*self.opt.crop_size
                new_down = float(down)/height*self.opt.crop_size
                if flip:
                    if w_padded:
                        data_instance['bbox'] = [int(self.opt.crop_size*3-new_right), int(new_up), int(self.opt.crop_size*3-new_left), int(new_down), w_padded, h_padded, width, height]
                    else:
                        data_instance['bbox'] = [int(self.opt.crop_size-new_right), int(new_up), int(self.opt.crop_size-new_left), int(new_down), w_padded, h_padded, width, height]
                else:
                    data_instance['bbox'] = [int(new_left), int(new_up), int(new_right), int(new_down), w_padded, h_padded, width, height]
                data_instance['image'] = transform_obj_image(cropped)
                data_instance['label'] = transform_obj_label(cropped_label) * 255.0
                if self.opt.use_depth:
                    depth_tensor_tmp = transform_obj_label(cropped_depth).float() * 255.0
                    data_instance['depth'] = ((depth_tensor_tmp-torch.min(depth_tensor_tmp))/(torch.max(depth_tensor_tmp)-torch.min(depth_tensor_tmp))-0.5) *2.0
                if self.opt.use_acgan:
                    object_class = self.mapping[obj_id]
                    object_tensor = torch.FloatTensor(1).fill_(object_class)
                    object_tensor = object_tensor.expand_as(data_instance['label'])
                    data_instance['object'] = object_tensor
                    data_instance['object_class'] = object_class
                if self.opt.use_scene:
                    scene_tensor = torch.FloatTensor(1).fill_(scene_class)
                    scene_tensor = scene_tensor.expand_as(data_instance['label'])
                    data_instance['scene'] = scene_tensor

                if self.opt.no_background:
                    data_instance['image'] = data_instance['image'] * data_instance['label'].long().float()
                if self.opt.real_background:
                    data_instance['fg'] = data_instance['image'] * data_instance['label'].long().float()
                    data_instance['bg'] = data_instance['image'] * (1-data_instance['label'].long()).float()
                if self.opt.position_encode:
                    _, h, w = data_instance['label'].size()
                    data_instance['pos_x'] = torch.LongTensor(np.array([[[j for j in range(w)] for i in range(h)]]))
                    data_instance['pos_y'] = torch.LongTensor(np.array([[[i for j in range(w)] for i in range(h)]]))
                data_instance['instance'] = 0
                all_objects[chosen_object] = data_instance

            for i,  chosen_object in enumerate(all_objects):
                all_objects[chosen_object]['object_name'] = chosen_object
                input_dict['object_%03d' % i] = all_objects[chosen_object]



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

        data_global = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }
        if self.opt.use_depth:
            data_global['depth'] = depth_tensor
#        if self.opt.position_encode:
#            _, h, w = label_tensor.size()
#            data_global['pos_x'] = torch.LongTensor(np.array([[[j for j in range(w)] for i in range(h)]]))
#            data_global['pos_y'] = torch.LongTensor(np.array([[[i for j in range(w)] for i in range(h)]]))
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
