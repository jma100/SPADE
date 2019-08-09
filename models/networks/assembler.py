import torch.nn.functional as F
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.base_network import BaseNetwork

class Assembler(BaseNetwork):
    '''Same architecture as the image discriminator'''
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.enhance_1 = SPADEResnetBlock(self.opt.output_nc, self.opt.output_nc, opt)
        self.enhance_2 = SPADEResnetBlock(self.opt.output_nc, self.opt.output_nc, opt)

    def forward(self, data):
        global_gen = data['global']['generated'].clone()
        global_label = data['global']['label']
        for obj, obj_data in data.items():
            if obj == 'global' or 'generated':
                continue
            instance_data = data[obj]
            left, up, right, down = [f.item() for f in instance_data['bbox']]
            instance_resized_gen = F.interpolate(instance_data['generated'], size=(right-left, down-up), mode='bilinear')
            instance_resized_mask = F.interpolate(instance_data['label'], size=(right-left, down-up), mode='nearest')
            global_gen[:, :, left:right, up:down] = data['global']['generated'][:, :, left:right, up:down] * (1-instance_resized_mask) + instance_resized_gen * instance_resized_mask
        global_gen = self.enhance_1(global_gen, global_label)
        global_gen = self.enhance_2(global_gen, global_label)
        return global_gen, data['global']['image'], global_label

