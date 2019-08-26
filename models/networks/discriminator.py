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
        if self.opt.use_acgan:
            class_result = []
        get_intermediate_features = not self.opt.no_ganFeat_loss
        for name, D in self.named_children():
            out = D(input)

            if self.opt.use_acgan:
                out, pred_class = D(input)
                class_result.append(pred_class)
            else:
                out = D(input)

            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        if self.opt.use_acgan:
            return result, class_result

        return result

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=padding, dilation=dilation)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride,
                             padding=dilation[0], dilation=dilation[0])
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride, 
                             padding=dilation[1], dilation=dilation[1])
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = self.relu(out)

        return out


class LocalDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, use_sigmoid):
        super(LocalDiscriminator, self).__init__()

        self.input_nc = input_nc
        self.ndf = ndf
        self.conv = nn.Conv2d
        self.batch_norm = nn.BatchNorm2d
        self.res_block = ResidualBlock

        self.model = self.create_discriminator(use_sigmoid)

    def create_discriminator(self, use_sigmoid):
        norm_layer = self.batch_norm
        ndf = self.ndf  # 32
        self.res_block = ResidualBlock
        
        sequence = [
            nn.Conv2d(self.input_nc, self.ndf, kernel_size=3, stride=2, padding=1),nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf, self.ndf * 4, kernel_size=3, stride=2, padding=1),nn.InstanceNorm2d(ndf* 4),
            nn.LeakyReLU(0.2, True),

            #nn.Conv2d(self.ndf * 2, self.ndf * 8, kernel_size=5, stride=2, padding=1),
            #nn.LeakyReLU(0.2, True),
            #nn.Dropout(0.2),
            
            self.res_block(self.ndf * 4, self.ndf * 4),
            self.res_block(self.ndf * 4, self.ndf * 4),

            nn.Conv2d(self.ndf * 4, self.ndf * 2, kernel_size=3, stride=2, padding=1), nn.InstanceNorm2d(ndf* 2),
            #nn.Dropout(0.2),

            nn.Conv2d(self.ndf * 2, 1, kernel_size=3, stride=2, padding=1)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        return nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

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

        self.aux_layer = nn.Conv2d(nf, opt.acgan_nc, kernel_size=kw, stride=1, padding=padw)

    def compute_D_input_nc(self, opt):
        input_nc = (1 if opt.is_object else opt.label_nc) + opt.output_nc
        if opt.contain_dontcare_label and not opt.is_object:
            input_nc += 1
        if not opt.no_instance:
            input_nc += 1
        if opt.use_depth:
            input_nc += 1
        if opt.use_normal:
            input_nc += 3
        if opt.use_material:
            input_nc += 1
        if opt.use_part:
            input_nc += 1
        if opt.use_acgan:
            input_nc += opt.acgan_nc
        return input_nc

    def forward(self, input):
        results = [input]
        num = len(list(self.children()))
        counter = 0
        for submodel in self.children():
            if counter == num-1:
                continue
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
            counter += 1

        if self.opt.use_acgan:
            pred_object = nn.Softmax(dim=1)(self.aux_layer(results[-2]))
            bs, c, h, w = pred_object.size()
            pred_object = pred_object.view(bs, c, h*w)

        get_intermediate_features = not self.opt.no_ganFeat_loss

        if self.opt.use_acgan:
            if get_intermediate_features:
                return results[1:], pred_object
            else:
                return results[-1], pred_object

        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
