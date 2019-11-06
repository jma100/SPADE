"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import models.networks as networks
import util.util as util


class Pix2PixModel(torch.nn.Module):
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

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain and not opt.load_pretrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if (opt.use_object_vae and opt.is_object) or (opt.use_stuff_vae and not opt.is_object):
                self.KLDLoss = networks.KLDLoss()
            if opt.use_acgan_loss:
                self.criterionACGAN = networks.ACLoss(
                tensor=self.FloatTensor, opt=self.opt)

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(data)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(data)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(data['image'])
            return z, mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, fake_features, z, _ = self.generate_fake(data)
            return fake_image, fake_features, z
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        postfix = '_object' if self.opt.is_object else '_global'
        util.save_network(self.netG, 'G'+postfix, epoch, self.opt)
        util.save_network(self.netD, 'D'+postfix, epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E'+postfix, epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain and not opt.load_pretrain else None
        netE = networks.define_E(opt) if (opt.use_object_vae and opt.is_object) or (opt.use_stuff_vae and not opt.is_object) else None

        if opt.load_pretrain or not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain and not opt.load_pretrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if (opt.use_object_vae and opt.is_object) or (opt.use_stuff_vae and not opt.is_object):
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE


    def compute_generator_loss(self, input_dict):
        G_losses = {}

        input_semantics, real_image = input_dict['label'], input_dict['image']
        fake_image, fake_features, KLD_loss = self.generate_fake(
            input_dict, compute_kld_loss=(self.opt.use_object_vae and self.opt.is_object) or (self.opt.use_stuff_vae and not self.opt.is_object))

        if (self.opt.use_object_vae and self.opt.is_object) or (self.opt.use_stuff_vae and not self.opt.is_object):
            G_losses['KLD'] = KLD_loss

        if self.opt.use_acgan_loss:
            pred_fake, pred_real, class_fake, class_real = self.discriminate(
                input_semantics, fake_image, real_image)
        else:
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if self.opt.use_acgan_loss:
            object_class = input_dict['object_class']
            G_losses['ACLoss'] = self.criterionACGAN(class_fake, object_class) * self.opt.lambda_acgan

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) \
                * self.opt.lambda_vgg

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_dict):
        D_losses = {}
        input_semantics, real_image = input_dict['label'], input_dict['image']
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_dict)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        if self.opt.use_acgan_loss:
            pred_fake, pred_real, class_fake, class_real = self.discriminate(
                input_semantics, fake_image, real_image)
        else:
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        if self.opt.use_acgan_loss:
            object_class = input_dict['object_class']
            D_losses['D_class_fake'] = self.criterionACGAN(class_fake, object_class) * self.opt.lambda_acgan
            D_losses['D_class_real'] = self.criterionACGAN(class_real, object_class) * self.opt.lambda_acgan

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_dict, compute_kld_loss=False):
        z = None
        KLD_loss = None
        real_image, input_semantics = input_dict['image'], input_dict['label']
        if (self.opt.use_object_vae and self.opt.is_object) or (self.opt.use_stuff_vae and not self.opt.is_object):
            if self.opt.real_background:
                encode_input = input_dict['fg']
            else:
                encode_input = real_image

            if self.opt.position_encode and self.opt.is_object:
                encode_input = torch.cat((encode_input, input_dict['pos_x'].float(), input_dict['pos_y'].float()), dim=1)

            z, mu, logvar = self.encode_z(encode_input)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image, fake_features  = self.netG(input_semantics, z=z)

        if self.opt.real_background:
            bg = input_dict['bg']
            fake_features = bg.cuda() + fake_features * (bg == 0).float().cuda()

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, fake_features, z, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        if self.opt.use_acgan_loss:
            discriminator_out, pred_class = self.netD(fake_and_real)
            pred_fake, pred_real = self.divide_pred(discriminator_out)
            class_fake, class_real = pred_class
            return pred_fake, pred_real, class_fake, class_real
        else:
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

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
