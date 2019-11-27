import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import scipy.io
import numpy as np
from PIL import Image
import os
from collections import Counter
import json
from tqdm import tqdm

def bbox(img, instance_id):
    a = np.where(img == instance_id)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    return bbox


class ADEGigasunDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(load_size=128)
        parser.set_defaults(crop_size=128)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=1)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        parser.add_argument('--train_list', type=str, help='import list of training folders')
        parser.add_argument('--gigasun_train_list', type=str, help='path to bunch of gigasun train txt files')
        parser.add_argument('--gigasun_upper_limit', type=int, help='object training data upper limit for each category, e.g. 20000')
        parser.add_argument('--use_scene', action='store_true', help='input scene category or not')
        parser.add_argument('--use_acgan_loss', action='store_true', help='add acgan loss or not')
        parser.set_defaults(use_acgan=True)
        parser.set_defaults(use_scene=True)
        parser.set_defaults(acgan_nc=88)
        parser.add_argument('--scene_nc', type=int, help='number of scene classes')
        parser.set_defaults(scene_nc=8)
        return parser

    def get_paths(self, opt):
        # Read ade training txt
        with open(opt.train_list,'r') as f:
            training_list = f.read().split('\n')
        if training_list[-1]=='':
            training_list = training_list[:-1]
        scene_paths = []
        image_paths = []
        label_paths = []
        for i,p in enumerate(training_list):
            if i % 3 == 0:
                scene_paths.append(p)
            elif i % 3 == 1:
                image_paths.append(p)
            elif i % 3 == 2:
                label_paths.append(p)

        instance_paths = [] # don't use instance map for ade20k
        ade_paths = {'label_paths': label_paths, 'image_paths': image_paths, 'instance_paths': instance_paths}
        if opt.use_scene:
            ade_paths['scene_paths'] = scene_paths

        # Read gigasun training txt
        if opt.gigasun_train_list:
            lines = open(opt.gigasun_train_list, 'r').read().split('\n')[:-1]
            image_paths = []
            label_paths = []
            json_paths = []
            category_ids = []
            mask_ids = []
            instance_paths = []
            isthings = []
            if opt.use_scene:
                scene_paths = []

            for line in lines:
                image_path, label_path, json_path, category_id, mask_id, isthing, scene = line.split()
                category_id = int(category_id)
                mask_id = int(mask_id)
                isthing = True if isthing == "True" else False
                image_paths.append(image_path)
                label_paths.append(label_path)
                json_paths.append(json_path)
                category_ids.append(category_id)
                mask_ids.append(mask_id)
                isthings.append(isthing)
                if opt.use_scene:
                    scene_paths.append(scene)
            
            gigasun_paths = {'image_paths': image_paths, 'label_paths': label_paths, 'json_paths': json_paths, 'category_ids': category_ids, 'mask_ids': mask_ids, 'instance_paths': instance_paths, 'isthings': isthings}
            if opt.use_scene:
                gigasun_paths['scene_paths'] = scene_paths
            return ade_paths, gigasun_paths

        return ade_paths


    ## In ADE20k, 'unknown' label is of value 0.
    ## Change the 'unknown' label to 255 to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc


    def crop_object(self, object_dict):
        thing_mapping = {'24': '116', '77': '109', '13': '70', '59': '8', '58': '126', '32': '120', '57': '24', '56': '20', '39': '99', '40': '148', '60': '16', '61': '66', '62': '90', '26': '116', '73': '68', '72': '51', '71': '48', '68': '125', '69': '119', '75': '136', '74': '149'}
        stuff_mapping = {'25': '25', '13': '37', '27': '54', '14': '28', '16': '58', '49': '121', '42': '16', '29': '82', '35': '64', '36': '9', '2': '132', '5': '46', '53': '29', '6': '19', '9': '67', '41': '11'}
        margin = 16
        label_path = object_dict['label_path']
        image_path = object_dict['image_path']
        mask_id = object_dict['mask_id']
        category_id = object_dict['category_id']
        isthing = object_dict['isthing']
        segm = Image.open(label_path)
        segm_data = np.array(segm)
        height, width = segm_data.shape
        img = Image.open(image_path).convert('RGB')
        if segm.size != img.size:
            img = img.resize(segm.size, resample=Image.BILINEAR)
        img_data = np.array(img)
        instance_masked = (segm_data == mask_id).astype('int')
        up, down, left, right = bbox(segm_data, mask_id)
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
                # depth_data_pad = np.pad(depth_data, ((height, height), (0,0)), 'reflect')
                # depth_new = Image.fromarray(depth_data_pad, mode='I')
                up += height
                down += height
                instance_masked_new = np.pad(instance_masked, ((height, height), (0,0)), 'constant')
            else:
                img_new = img
                instance_masked_new = instance_masked
                # depth_new = depth
                
        elif h > w:
            # Make square
            center = (left+right)//2
            left = center - h//2
            right = center + (h-h//2)
            assert right-left == down-up
            if left < 0 or right > height:
                img_data_pad = np.pad(img_data, ((0,0), (width, width), (0,0)), 'reflect')
                img_new = Image.fromarray(img_data_pad)
                # depth_data_pad = np.pad(depth_data, ((0,0), (width, width)), 'reflect')
                # depth_new = Image.fromarray(depth_data_pad, mode='I')
                left += width
                right += width
                instance_masked_new = np.pad(instance_masked, ((0,0), (width, width)), 'constant')
            else:
                img_new = img
                instance_masked_new = instance_masked
                # depth_new = depth
        else:
            img_new = img
            instance_masked_new = instance_masked
            # depth_new = depth

        assert down-up == right-left
        # cropped_depth = depth_new.crop((left, up, right, down))
        cropped = img_new.crop((left, up, right, down))
        cropped_mask = instance_masked_new[up:down, left:right]
        cropped_label = Image.fromarray(cropped_mask.astype('uint8'))
        if isthing:
            obj_class = int(thing_mapping[str(category_id)])
        else:
            obj_class = int(stuff_mapping[str(category_id)])

        return cropped, cropped_label, obj_class



