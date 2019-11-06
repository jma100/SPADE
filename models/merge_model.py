import torch

from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
import models.networks as networks
from models.pix2pix_model import Pix2PixModel
import util.util as util

class MergeModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        opt.is_object = True
        self.net_object = Pix2PixModel(opt)
        opt.is_object = False
        self.net_global = Pix2PixModel(opt)
        self.netA, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)

    def initialize_networks(self, opt):
        netA = networks.define_A(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netA = util.load_network(netA, 'A', opt.which_epoch, opt, is_pix2pix=False)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt, is_pix2pix=False)

        return netA, netD

    def forward(self, data, mode=None):
        if mode == 'assemble':
            a_loss, generated = self.compute_assembler_loss(data)
            return a_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(data)
            return d_loss
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _, _ = self.netA(data)
            return fake_image
        # elif mode == 'object_generator':
        #     return 
        # elif mode == 'object_discriminator':
        # elif mode == 'global_generator':
        # elif mode == 'global_discriminator':
        else:
            raise ValueError("|mode| is invalid")


    def compute_assembler_loss(self, data):
        G_losses = {}

        fake_image, real_image, input_semantics = self.netA(data)

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image       

    def discriminate(self, input_semantics, fake, real): 
        fake_concat = torch.cat([input_semantics, fake], dim=1)
        real_concat = torch.cat([input_semantics, real], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def compute_discriminator_loss(self, data):
        D_losses = {}
        with torch.no_grad():
            fake_image, real_image, input_semantics = self.netA(data)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.opt.use_acgan and self.opt.is_object:
            data['object'] = data['object'].long()
        if self.opt.use_scene and self.opt.is_object:
            data['scene'] = data['scene'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()
            if self.opt.use_depth:
                data['depth'] = data['depth'].cuda()
            if self.opt.real_background:
                data['fg'] = data['fg'].cuda()
                data['bg'] = data['bg'].cuda()
            if self.opt.use_acgan and self.opt.is_object:
                data['object'] = data['object'].cuda()
            if self.opt.use_scene and self.opt.is_object:
                data['scene'] = data['scene'].cuda()
            if self.opt.position_encode and self.opt.is_object:
                data['pos_x'] = data['pos_x'].cuda()
                data['pos_y'] = data['pos_y'].cuda()


        label_map = data['label']
        bs, _, h, w = label_map.size()
        if self.opt.is_object:
            nc = 1+1 if self.opt.contain_dontcare_label else 1
        else:
            nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
                else self.opt.label_nc
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        if self.opt.use_depth:
            input_semantics = torch.cat((input_semantics, data['depth']),dim=1)

        # create one-hot object label
        if self.opt.use_acgan and self.opt.is_object:
            object_map = data['object']
            input_object = self.FloatTensor(bs, self.opt.acgan_nc, h, w).zero_()
            input_object_map = input_object.scatter_(1, object_map, 1.0)
            input_semantics = torch.cat((input_semantics, input_object_map), dim=1)

        # create one-hot scene label
        if self.opt.use_scene and self.opt.is_object:
            scene_map = data['scene']
            input_scene = self.FloatTensor(bs, self.opt.scene_nc, h, w).zero_()
            input_scene_map = input_scene.scatter_(1, scene_map, 1.0)
            input_semantics = torch.cat((input_semantics, input_scene_map), dim=1)

        input_dict = {'label': input_semantics, 'image': data['image']}
        if not self.opt.is_object:
            input_dict['path'] = data['path']
        if self.opt.use_acgan and self.opt.is_object:
            input_dict['object_class'] = data['object_class']
        if self.opt.real_background:
            input_dict['fg'] = data['fg']
            input_dict['bg'] = data['bg']
        if self.opt.position_encode and self.opt.is_object:
            input_dict['pos_x'] = data['pos_x']
            input_dict['pos_y'] = data['pos_y']
        if 'generated' in data:
            input_dict['generated'] = data['generated']
        if self.opt.is_object:
            input_dict['bbox'] = data['bbox']
            input_dict['object_name'] = data['object_name']
            input_dict['mask'] = data['label'].float()
        return input_dict

    def create_optimizers(self, opt):
        if opt.load_pretrain:
            A_params = list(self.netA.parameters())
        else:
            A_params = list(self.netA.parameters())+ \
                    list(self.net_object.netG.parameters()) + \
                    list(self.net_global.netG.parameters())
            if opt.use_object_vae:
                A_params += list(self.net_object.netE.parameters())
            if opt.use_stuff_vae:
                A_params += list(self.net_global.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            A_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            A_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_A = torch.optim.Adam(A_params, lr=A_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_A, optimizer_D

    def save(self, epoch):
        util.save_network(self.netA, 'A', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if not self.opt.load_pretrain:
            self.net_object.save(epoch)
            self.net_global.save(epoch)

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
