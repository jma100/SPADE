import torch.nn.functional as F
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.base_network import BaseNetwork
import torch.nn as nn

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
        self.conv_merge = nn.Conv2d(2 * nf, 1 * nf, 1)

    def forward(self, data):
        z = data['z']
        global_gen = data['global']['features'].clone()
        _, _, height, width = global_gen.size()
        global_label = F.interpolate(data['global']['label'], size=(height, width), mode='nearest')
        object_layer = data['global']['features'].clone()
        for i in range(global_gen.size()[0]):
            for obj, obj_data in data.items():
                if obj in ['global', 'z', 'generated']:
                    continue
                instance_data = data[obj]
                left, up, right, down = [f[i].item() for f in instance_data['bbox']]
#                print('----------------assembler-------------')
#                print(left, up, right, down)
#                print(data['global']['path'][i])
                instance_resized_gen = F.interpolate(instance_data['features'][i:i+1,:,:,:], size=(down-up, right-left), mode='bilinear')
                instance_resized_mask = F.interpolate(instance_data['label'][i:i+1, :, :, :], size=(down-up, right-left), mode='nearest')
                object_layer[i:i+1, :, up:down, left:right] = object_layer[i:i+1, :, up:down, left:right] * (1-instance_resized_mask) + instance_resized_gen * instance_resized_mask
        global_gen = torch.cat((global_gen, object_layer), dim=1)
        global_gen = self.conv_merge(F.leaky_relu(global_gen, 2e-1, inplace=True))
        global_gen = F.leaky_relu(global_gen, 2e-1, inplace=True)
        global_gen = self.enhance_1(global_gen, data['global']['label'], z)
        global_gen = self.enhance_2(global_gen, data['global']['label'], z)
        global_gen = self.conv_img(F.leaky_relu(global_gen, 2e-1, inplace=True))
        global_gen = F.tanh(global_gen)
        return global_gen, data['global']['image'], data['global']['label']

