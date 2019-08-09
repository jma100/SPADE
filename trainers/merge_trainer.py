from models.networks.sync_batchnorm import DataParallelWithCallback
from models.merge_model import MergeModel
from models.pix2pix_model import Pix2PixModel


class MergeTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.merge_model = MergeModel(opt) # implement
        
        if len(opt.gpu_ids) > 0:
            self.merge_model = DataParallelWithCallback(self.merge_model,
                                                          device_ids=opt.gpu_ids)
            self.merge_model_on_one_gpu = self.merge_model.module
        else:
            self.merge_model_on_one_gpu = self.merge_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_A, self.optimizer_D = \
                self.merge_model_on_one_gpu.create_optimizers(opt)
            self.optimizer_G_object, self.optimizer_D_object = \
                self.merge_model_on_one_gpu.net_object.create_optimizers(opt)
            self.optimizer_G_global, self.optimizer_D_global = \
                self.merge_model_on_one_gpu.net_global.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_overall_one_step(self, data, object_generated, global_generated):
        self.optimizer_G_object.zero_grad()
        self.optimizer_G_global.zero_grad()
        self.optimizer_A.zero_grad()
        a_losses, generated = self.merge_model(data, mode='assemble')
        loss = sum(a_losses.values()).mean()
        loss.backward()
        self.optimizer_G_object.step()
        self.optimizer_G_global.step()
        self.optimizer_A.step()
        self.a_losses = a_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.merge_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def run_object_generator_one_step(self, data):
        self.optimizer_G_object.zero_grad()
        object_g_losses, object_generated = self.merge_model.module.net_object(data, mode='generator')
        object_g_loss = sum(object_g_losses.values()).mean()
        object_g_loss.backward(retain_graph=True)
#        object_g_loss.backward()
        self.optimizer_G_object.step()
        self.object_g_losses = object_g_losses
        self.object_generated = object_generated

    def run_object_discriminator_one_step(self, data):
        self.optimizer_D_object.zero_grad()
        object_d_losses = self.merge_model.module.net_object(data, mode='discriminator')
        object_d_loss = sum(object_d_losses.values()).mean()
        object_d_loss.backward()
        self.optimizer_D_object.step()
        self.object_d_losses = object_d_losses

    def run_global_generator_one_step(self, data):
        self.optimizer_G_global.zero_grad()
        global_g_losses, global_generated = self.merge_model.module.net_global(data, mode='generator')
        global_g_loss = sum(global_g_losses.values()).mean()
        global_g_loss.backward(retain_graph=True)
#        global_g_loss.backward()
        self.optimizer_G_global.step()
        self.global_g_losses = global_g_losses
        self.global_generated = global_generated

    def run_global_discriminator_one_step(self, data):
        self.optimizer_D_global.zero_grad()
        global_d_losses = self.merge_model.module.net_global(data, mode='discriminator')
        global_d_loss = sum(global_d_losses.values()).mean()
        global_d_loss.backward()
        self.optimizer_D_global.step()
        self.global_d_losses = global_d_losses

    def get_latest_losses(self):
        return [{**self.a_losses, **self.d_losses}, 
                {**self.object_g_losses, **self.object_d_losses}, 
                {**self.global_g_losses, **self.global_d_losses}]

    def get_latest_generated(self):
        return self.generated, self.object_generated, self.global_generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.merge_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_A.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

