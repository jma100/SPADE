"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()
        if opt.is_object:
            self.encode_size = opt.obj_crop_size
        else:
            self.encode_size = opt.crop_size

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))

        if (opt.is_object and opt.obj_crop_size >=64) or (not opt.is_object and opt.crop_size >= 64):
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, kw, stride=2, padding=pw))
        if (opt.is_object and opt.obj_crop_size >=128) or (not opt.is_object and opt.crop_size >= 128):
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
            self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if (opt.is_object and opt.obj_crop_size >=256) or (not opt.is_object and opt.crop_size >= 256):
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        if (opt.is_object and opt.obj_crop_size >=128) or (not opt.is_object and opt.crop_size >= 128):
            self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, opt.z_dim)
            self.fc_var = nn.Linear(ndf * 8 * s0 * s0, opt.z_dim)
        else:
            self.fc_mu = nn.Linear(ndf * 4 * s0 * s0, opt.z_dim)
            self.fc_var = nn.Linear(ndf * 4 * s0 * s0, opt.z_dim)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != self.encode_size or x.size(3) != self.encode_size:
            x = F.interpolate(x, size=(self.encode_size, self.encode_size), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        if (self.opt.is_object and self.opt.obj_crop_size >=128) or (not self.opt.is_object and self.opt.crop_size >= 128):
            x = self.layer5(self.actvn(x))
        if (self.opt.is_object and self.opt.obj_crop_size >=256) or (not self.opt.is_object and self.opt.crop_size >= 256):
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar

