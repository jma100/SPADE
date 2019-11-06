"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
import util.util as util


class MultiscaleDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--netD_subarch', type=str, default='n_layer',
                            help='architecture of each discriminator')
        parser.add_argument('--num_D', type=int, default=2,
                            help='number of discriminators to be used in multiscale')
        opt, _ = parser.parse_known_args()

        # define properties of each discriminator of the multiscale discriminator
        subnetD = util.find_class_in_module(opt.netD_subarch + 'discriminator',
                                            'models.networks.discriminator')
        subnetD.modify_commandline_options(parser, is_train)

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        for i in range(opt.num_D):
            subnetD = self.create_single_discriminator(opt)
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self, opt):
        subarch = opt.netD_subarch
        if subarch == 'n_layer':
            netD = NLayerDiscriminator(opt)
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        if self.opt.use_acgan_loss or (self.opt.is_object and self.opt.acgan_nc > 1):
            class_result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)

            if self.opt.use_acgan_loss or (self.opt.is_object and self.opt.acgan_nc > 1):
                out, pred_class = D(input)
                class_result.append(pred_class)
            else:
                out = D(input)

            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        if self.opt.use_acgan_loss or (self.opt.is_object and self.opt.acgan_nc > 1):
            return result, class_result

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--n_layers_D', type=int, default=4,
                            help='# layers in each discriminator')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ndf
        input_nc = self.compute_D_input_nc(opt)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_D)
        sequence = [[nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, opt.n_layers_D):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == opt.n_layers_D - 1 else 2
            sequence += [[norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

        if self.opt.use_acgan_loss:
            self.aux_layer = nn.Conv2d(nf, opt.acgan_nc, kernel_size=kw, stride=1, padding=padw)

    def compute_D_input_nc(self, opt):
        input_nc = (1 if opt.is_object else opt.label_nc) + opt.output_nc
        if opt.contain_dontcare_label and not opt.is_object:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if opt.use_depth:
            input_nc += 1
        if opt.use_acgan and opt.is_object:
            input_nc += opt.acgan_nc
        return input_nc

    def forward(self, input):
        results = [input]
        num = len(list(self.children()))
        counter = 0
        for submodel in self.children():
            if self.opt.use_acgan and counter == num-1:
                continue
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
            counter += 1

        if self.opt.use_acgan_loss:
            pred_object = nn.Softmax(dim=1)(self.aux_layer(results[-2]))
            bs, c, h, w = pred_object.size()
            pred_object = pred_object.view(bs, c, h*w)

        get_intermediate_features = not self.opt.no_ganFeat_loss

        if self.opt.use_acgan_loss:
            if get_intermediate_features:
                return results[1:], pred_object
            else:
                return results[-1], pred_object

        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
