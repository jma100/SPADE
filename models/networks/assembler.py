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
        self.enhance_1 = SPADEResnetBlock(self.opt.output_nc, self.opt.output_nc, opt)
        self.enhance_2 = SPADEResnetBlock(self.opt.output_nc, self.opt.output_nc, opt)
        self.conv_img = nn.Conv2d(final_nc, self.opt.output_nc, 3, padding=1)


    def forward(self, data):
        global_gen = data['global']['generated'].clone()
        global_label = data['global']['label']
        for i in range(global_gen.size()[0]):
            for obj, obj_data in data.items():
                if obj == 'global' or obj == 'generated':
                    continue
                instance_data = data[obj]
                left, up, right, down = [f[i].item() for f in instance_data['bbox']]
#                print('----------------assembler-------------')
#                print(left, up, right, down)
#                print(data['global']['path'][i])
                instance_resized_gen = F.interpolate(instance_data['generated'][i:i+1,:,:,:], size=(down-up, right-left), mode='bilinear')
                instance_resized_mask = F.interpolate(instance_data['label'][i:i+1, :, :, :], size=(down-up, right-left), mode='nearest')
                global_gen[i:i+1, :, up:down, left:right] = data['global']['generated'][i:i+1, :, up:down, left:right] * (1-instance_resized_mask) + instance_resized_gen * instance_resized_mask
        global_gen = self.conv_img(F.leaky_relu(global_gen, 2e-1, inplace=True))
        global_gen = F.tanh(global_gen)
        global_gen = self.enhance_1(global_gen, data['global']['label'])
        global_gen = self.enhance_2(global_gen, data['global']['label'])
        return global_gen, data['global']['image'], data['global']['label']

