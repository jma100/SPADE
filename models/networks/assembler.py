import torch.nn.functional as F
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.base_network import BaseNetwork
import torch.nn as nn
import torch

class Assembler(BaseNetwork):
    '''Same architecture as the image discriminator'''
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        nf = opt.ngf
        final_nc = nf
        self.enhance_1 = SPADEResnetBlock(1 * nf, 1 * nf, opt, assemble=True)
        self.enhance_2 = SPADEResnetBlock(1 * nf, 1 * nf, opt, assemble=True)
        self.conv_img = nn.Conv2d(final_nc, self.opt.output_nc, 3, padding=1)
        if not self.opt.no_merge_layer:
            self.conv_merge = nn.Conv2d(2 * nf, 1 * nf, 1)

    def forward(self, data):
        z = data['z']
        global_gen = data['global']['features'].clone()
        _, _, height, width = global_gen.size()
        global_label = F.interpolate(data['global']['label'], size=(height, width), mode='nearest')
        for i in range(global_gen.size()[0]):
            for obj, obj_data in data.items():
                if obj in ['global', 'z', 'generated']:
                    continue
                instance_data = data[obj]
                left, up, right, down, w_padded, h_padded, w, h = [f[i].item() for f in instance_data['bbox']]
                size = self.opt.crop_size
                instance_resized_gen = F.interpolate(instance_data['features'][i:i+1,:,:,:], size=(down-up, right-left), mode='bilinear')
                instance_resized_mask = F.interpolate(instance_data['mask'][i:i+1, :, :, :], size=(down-up, right-left), mode='nearest')


                # find left, right, up, down coordinates in the 256 by 256
                c_left = left
                c_right = right
                c_up = up
                c_down = down

                if w_padded:
                    # if left coordinate -crop_size out of boundary
                    new_left = left - self.opt.crop_size
                    c_left = max(0, new_left)
                    obj_c_left = c_left-new_left
                    new_right = right - self.opt.crop_size
                    c_right = min(new_right, self.opt.crop_size)
                    obj_c_right = right-left-(new_right-c_right)
                    instance_resized_gen = instance_resized_gen[:, :, :, obj_c_left:obj_c_right]
                    instance_resized_mask = instance_resized_mask[:, :, :, obj_c_left:obj_c_right]
                if h_padded:
                    new_up = up-self.opt.crop_size
                    c_up = max(0, new_up)
                    obj_c_up = c_up-new_up
                    new_down = down - self.opt.crop_size
                    c_down = min(new_down, self.opt.crop_size)
                    obj_c_down = down-up-(new_down-c_down)
                    instance_resized_gen = instance_resized_gen[:, :, obj_c_up:obj_c_down, :]
                    instance_resized_mask = instance_resized_mask[:, :, obj_c_up:obj_c_down, :]
                assert not (w_padded and h_padded)
                global_gen[i:i+1, :, c_up:c_down, c_left:c_right] = global_gen[i:i+1, :, c_up:c_down, c_left:c_right].clone() * (1-instance_resized_mask) + instance_resized_gen * instance_resized_mask
        if not self.opt.no_merge_layer:
            global_gen = torch.cat((data['global']['features'], global_gen), dim=1)
            global_gen = self.conv_merge(global_gen)
#        global_gen = self.conv_merge(F.leaky_relu(global_gen, 2e-1, inplace=True))
#        global_gen = F.leaky_relu(global_gen, 2e-1, inplace=True)
        global_gen = self.enhance_1(global_gen, data['global']['label'], z)
        global_gen = self.enhance_2(global_gen, data['global']['label'], z)
        global_gen = self.conv_img(F.leaky_relu(global_gen, 2e-1, inplace=True))
        global_gen = F.tanh(global_gen)
        return global_gen, data['global']['image'], data['global']['label']

