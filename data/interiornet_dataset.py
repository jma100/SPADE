import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import scipy.io
import numpy as np
from PIL import Image

class InteriornetDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=40)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        parser.add_argument('--train_list', type=str, help='import list of training folders')
        return parser

    def get_paths(self, opt):
        with open(opt.train_list,'r') as f:
            training_list = f.read().split('\n')
        if training_list[-1]=='':
            training_list = training_list[:-1]
        image_paths = []
        label_paths = []
        depth_paths = []
        illumination_paths = []
        material_paths = []
        for i,p in enumerate(training_list):
            if 'cam0' in p or 'normal' in p:
                image_paths.append(p)
            elif 'label' in p:
                label_paths.append(p)
            elif 'illumination' in p:
                illumination_paths.append(p)
            elif opt.use_depth and 'depth' in p:
                depth_paths.append(p)


        instance_paths = [] # don't use instance map for ade20k
        return label_paths, image_paths, instance_paths, depth_paths, material_paths, illumination_paths
#        if opt.use_illumination and opt.use_depth:
#            return label_paths, image_paths, instance_paths, depth_paths, illumination_paths
#        if opt.use_depth:
#            return label_paths, image_paths, instance_paths, depth_paths
#        else:
#            return label_paths, image_paths, instance_paths

    ## In ADE20k, 'unknown' label is of value 0.
    ## Change the 'unknown' label to 255 to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc

