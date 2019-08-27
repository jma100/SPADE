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
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()
            if opt.use_acgan:
                self.criterionACGAN = networks.ACLoss(
                tensor=self.FloatTensor, opt=self.opt)
            if opt.use_l1:
                self.criterionL1 = torch.nn.L1Loss()
            if opt.use_cyclez:
                self.criterionCycle = torch.nn.L1Loss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_dict = self.preprocess_input(data)
        input_semantics, real_image = input_dict['label'], input_dict['image']

        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, input_dict)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, input_dict)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return z, mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _, _ = self.generate_fake(input_semantics, real_image, input_dict)
            return fake_image
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
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.opt.use_acgan:
            data['object'] = data['object'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()
            if self.opt.use_depth:
                data['depth'] = data['depth'].cuda()
            if self.opt.use_normal:
                data['normal'] = data['normal'].cuda()
            if self.opt.use_material:
                data['material'] = data['material'].cuda()
            if self.opt.use_part:
                data['part'] = data['part'].cuda()
            if self.opt.real_background:
                data['fg'] = data['fg'].cuda()
                data['bg'] = data['bg'].cuda()
            if self.opt.use_acgan:
                data['object'] = data['object'].cuda()
            if self.opt.position_encode:
                data['pos_x'] = data['pos_x'].cuda()
                data['pos_y'] = data['pos_y'].cuda()
            if self.opt.use_image != '':
                data['encode'] = data['encode'].cuda()
        if self.opt.is_object and not self.opt.old_version:
            input_semantics = data['label'].float()
        else:
            # create one-hot label map
            label_map = data['label']
            bs, _, h, w = label_map.size()
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
        if self.opt.use_normal:
            input_semantics = torch.cat((input_semantics, data['normal']),dim=1)
        if self.opt.use_material:
            input_semantics = torch.cat((input_semantics, data['material']),dim=1)
        if self.opt.use_part:
            input_semantics = torch.cat((input_semantics, data['part']),dim=1)    

        # create one-hot object label
        if self.opt.use_acgan:
            object_map = data['object']
            input_object = self.FloatTensor(bs, self.opt.acgan_nc, h, w).zero_()
            input_object_map = input_object.scatter_(1, object_map, 1.0)
            input_semantics = torch.cat((input_semantics, input_object_map), dim=1)

        input_dict = {'label': input_semantics, 'image': data['image']}
        if self.opt.use_acgan:
            input_dict['object_class'] = data['object_class']
        if self.opt.real_background:
            input_dict['fg'] = data['fg']
            input_dict['bg'] = data['bg']
        if self.opt.position_encode:
            input_dict['pos_x'] = data['pos_x']
            input_dict['pos_y'] = data['pos_y']
        return input_dict

    def compute_generator_loss(self, input_semantics, real_image, input_dict):
        G_losses = {}


        fake_image, KLD_loss, CycleZ_loss = self.generate_fake(
            input_semantics, real_image, input_dict, compute_kld_loss=self.opt.use_vae, compute_cyclez_loss=self.opt.use_cyclez)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        if self.opt.use_acgan:
            pred_fake, pred_real, class_fake, class_real = self.discriminate(
                input_semantics, fake_image, real_image)
        else:
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if self.opt.use_l1:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) * self.opt.lambda_l1

        if self.opt.use_cyclez:
            G_losses['CycleZ'] = CycleZ_loss

        if self.opt.use_acgan:
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

    def compute_discriminator_loss(self, input_semantics, real_image, input_dict):
        D_losses = {}
        with torch.no_grad():
            fake_image, _, _ = self.generate_fake(input_semantics, real_image, input_dict)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        if self.opt.use_acgan:
            pred_fake, pred_real, class_fake, class_real = self.discriminate(
                input_semantics, fake_image, real_image)
        else:
            pred_fake, pred_real = self.discriminate(
                input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        if self.opt.use_acgan:
            object_class = input_dict['object_class']
            D_losses['D_class_fake'] = self.criterionACGAN(class_fake, object_class) * self.opt.lambda_acgan
            D_losses['D_class_real'] = self.criterionACGAN(class_real, object_class) * self.opt.lambda_acgan

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, input_dict, compute_kld_loss=False, compute_cyclez_loss=False):
        z = None
        KLD_loss = None
        CycleZ_loss = None
        if self.opt.use_vae:
            if self.opt.use_image != '':
                z, mu, logvar = self.encode_z(input_dict['encode'])
                print('encoded')
            elif self.opt.real_background:
                encode_input = input_dict['fg']
            else:
                encode_input = real_image
            if self.opt.position_encode:
                encode_input = torch.cat((encode_input, input_dict['pos_x'].float(), input_dict['pos_y'].float()), dim=1)
            z, mu, logvar = self.encode_z(encode_input)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image = self.netG(input_semantics, z=z)

        if self.opt.real_background:
            bg = input_dict['bg']
            fake_image = bg.cuda() + fake_image * (bg == 0).float().cuda()

        if compute_cyclez_loss:
            fake_z, fake_mu, fake_logvar = self.encode_z(fake_image)
            CycleZ_loss = (self.criterionCycle(fake_mu, mu) + self.criterionCycle(fake_logvar, logvar)) * self.opt.lambda_cyclez

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss, CycleZ_loss

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

        if self.opt.use_acgan:
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
