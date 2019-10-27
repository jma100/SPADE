import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import scipy.io
import numpy as np
from PIL import Image

class ACGanDataset(Pix2pixDataset):

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
        parser.add_argument('--train_list', type=str, help='import list of training folders')
#        parser.add_argument('--mapping_path', type=str, help='mapping dictionary for object categories')
        parser.set_defaults(use_acgan=True)
        parser.add_argument('--use_scene', type=str, default=True, help='use scene category as an input')
        return parser

    def get_paths(self, opt):
        assert opt.use_scene
        with open(opt.train_list,'r') as f:
            training_list = f.read().split('\n')[:-1]
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
        return label_paths, image_paths, scene_paths, instance_paths

    ## In ADE20k, 'unknown' label is of value 0.
    ## Change the 'unknown' label to 255 to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc

