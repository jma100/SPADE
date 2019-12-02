"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.nn.utils.spectral_norm as spectral_norm
from models.networks.normalization import SPADE


# ResNet block that uses SPADE.
# It differs from the ResNet block of pix2pixHD in that
# it takes in the segmentation map as input, learns the skip connection if necessary,
# and applies normalization first and then convolution.
# This architecture seemed like a standard architecture for unconditional or
# class-conditional GAN architecture using residual block.
# The code was inspired from https://github.com/LMescheder/GAN_stability.
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, assemble=False):
        super().__init__()
        # Attributes
        self.opt = opt
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        #input_nc = (1 if opt.is_object else opt.label_nc) + (1 if opt.contain_dontcare_label and not opt.is_object else 0) + (0 if opt.no_instance else 1) + (1 if opt.use_depth else 0) + (opt.acgan_nc if opt.use_acgan else 0)
        input_nc = (1 if opt.is_object else opt.label_nc) + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1) + (1 if opt.use_depth else 0) + (opt.acgan_nc if opt.use_acgan and opt.is_object else 0) + (opt.scene_nc if opt.use_scene and opt.is_object else 0)

        if opt.use_style and assemble and opt.use_stuff_vae and opt.use_object_vae:
            input_nc += opt.max_object_per_image + 1
        elif (opt.use_object_z and opt.is_object) or (opt.use_style and assemble and opt.use_object_vae):
            input_nc += opt.max_object_per_image
        elif (opt.use_stuff_z and not opt.is_object) or (opt.use_style and assemble and opt.use_stuff_vae):
            input_nc += 1

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        self.norm_0 = SPADE(spade_config_str, fin, input_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, input_nc)
        self.assemble = assemble
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, input_nc)
        if assemble and opt.use_stuff_vae and opt.use_object_vae:
            self.fc = nn.Linear(opt.z_dim*(opt.max_object_per_image+1), opt.w_dim*(opt.max_object_per_image+1))
        elif assemble and opt.use_object_vae:
            self.fc = nn.Linear(opt.z_dim*opt.max_object_per_image, opt.w_dim*opt.max_object_per_image)
        elif assemble and opt.use_stuff_vae:
            self.fc = nn.Linear(opt.z_dim, opt.w_dim)
        elif (opt.use_object_z and opt.is_object) or (opt.use_stuff_z and not opt.is_object and not assemble):
            self.fc = nn.Linear(opt.z_dim, opt.w_dim)
        self.w_dim = opt.w_dim

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg, z=None):
        if z is not None:
            w = self.fc(z)
            w = w.view(x.size()[0], -1, int(self.w_dim**0.5), int(self.w_dim**0.5))
            x_s = self.shortcut_with_style(x, seg, w)

            dx = self.conv_0(self.actvn(self.norm_0(x, seg, w)))
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg, w)))

            out = x_s + dx

            return out
        else:
            x_s = self.shortcut(x, seg)

            dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

            out = x_s + dx

            return out

    def shortcut_with_style(self, x, seg, w):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, seg, w)))
        else:
            x_s = x
        return x_s

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.actvn(self.norm_s(x, seg)))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1, inplace=True)


# ResNet block used in pix2pixHD
# We keep the same architecture as pix2pixHD.
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, activation=nn.ReLU(False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size)),
            activation,
            nn.ReflectionPad2d(pw),
            norm_layer(nn.Conv2d(dim, dim, kernel_size=kernel_size))
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return out


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
