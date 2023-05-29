import torch
import numpy as np
import random
import time
import os
import matplotlib.pyplot as plt


noEMA = True


class image_saver():
    def __init__(self):
        self.cols = 4
        self.rows = 3
        self.grid = 5
        self.checkpoints_dir = 'checkpoints'
        self.name = 'bile_duct'
        self.path = os.path.join(self.checkpoints_dir, self.name, "images")+"/"
        self.num_cl = 2
        self.no_EMA = noEMA
        os.makedirs(self.path, exist_ok=True)

    def visualize_batch(self, model, image, label, cur_iter):
        self.save_images(label, "label", cur_iter, is_label=True)
        self.save_images(image, "real", cur_iter)
        with torch.no_grad():
            model.eval()
            fake = model.module.netG(label)
            self.save_images(fake, "fake", cur_iter)
            model.train()
            if not self.no_EMA:
                model.eval()
                fake = model.module.netEMA(label)
                self.save_images(fake, "fake_ema", cur_iter)
                model.train()

    def save_images(self, batch, name, cur_iter, is_label=False):
        fig = plt.figure()
        for i in range(min(self.rows * self.cols, len(batch))):
            if is_label:
                im = tens_to_lab(batch[i], self.num_cl)
            else:
                im = tens_to_im(batch[i])
            plt.axis("off")
            fig.add_subplot(self.rows, self.cols, i+1)
            plt.axis("off")
            plt.imshow(im)
        fig.tight_layout()
        plt.savefig(self.path+str(cur_iter)+"_"+name)
        plt.close()

def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))

def tens_to_lab(tens, num_cl):
    label_tensor = Colorize(tens, num_cl)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    return label_numpy

def Colorize(tens, num_cl):
    cmap = labelcolormap(num_cl)
    cmap = torch.from_numpy(cmap[:num_cl])
    size = tens.size()
    color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
    tens = torch.argmax(tens, dim=0, keepdim=True)

    for label in range(0, len(cmap)):
        mask = (label == tens[0]).cpu()
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image

def labelcolormap(N):
    if N == 35:
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i + 1  # let's give 0 a color
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


class losses_saver():
    def __init__(self):
        self.name_list = ["Generator", "Vgg", "D_fake", "D_real", "LabelMix"]
        # self.name_list = ["Generator", "D_fake", "D_real", "LabelMix"]
        self.freq_smooth_loss = 50
        self.freq_save_loss = 200
        self.losses = dict()
        self.cur_estimates = np.zeros(len(self.name_list))
        self.continue_train = False

        self.checkpoints_dir = 'checkpoints'
        self.name = 'bile_duct'

        self.path = os.path.join(self.checkpoints_dir, self.name, "losses")
        self.is_first = True

        os.makedirs(self.path, exist_ok=True)
        for name in self.name_list:
            if self.continue_train:
                self.losses[name] = np.load(self.path+"/losses.npy", allow_pickle = True).item()[name]
            else:
                self.losses[name] = list()

    def __call__(self, epoch, losses):
        for i, loss in enumerate(losses):
            if loss is None:
                self.cur_estimates[i] = None
            else:
                self.cur_estimates[i] += loss.detach().cpu().numpy()
        if epoch % self.freq_smooth_loss == self.freq_smooth_loss-1:
            for i, loss in enumerate(losses):
                if not self.cur_estimates[i] is None:
                    self.losses[self.name_list[i]].append(self.cur_estimates[i]/self.freq_smooth_loss)
                    self.cur_estimates[i] = 0
        if epoch % self.freq_save_loss == self.freq_save_loss-1:
            self.plot_losses()
            np.save(os.path.join(self.checkpoints_dir, self.name, 'losses', 'losses'), self.losses)

    def plot_losses(self):
        for curve in self.losses:
            fig,ax = plt.subplots(1)
            n = np.array(range(len(self.losses[curve])))*self.freq_smooth_loss
            plt.plot(n[1:], self.losses[curve][1:])
            plt.ylabel('loss')
            plt.xlabel('iters')

            plt.savefig(os.path.join(self.checkpoints_dir, self.name, 'losses', '%s.png' % (curve)),  dpi=600)
            plt.close(fig)

        fig,ax = plt.subplots(1)
        for curve in self.losses:
            if np.isnan(self.losses[curve][0]):
                continue
            plt.plot(n[1:], self.losses[curve][1:], label=curve)
        plt.ylabel('loss')
        plt.xlabel('iters')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.checkpoints_dir, 'bile_duct', 'losses', 'combined.png'), dpi=600)
        plt.close(fig)



def save_networks(cur_iter, model, latest=False, best=False):
    checkpoints_dir = 'checkpoints'
    name = 'bile_duct'
    no_EMA = noEMA

    path = os.path.join(checkpoints_dir, name, "models")
    os.makedirs(path, exist_ok=True)
    if latest:
        torch.save(model.module.netG.state_dict(), path+'/%s_G.pth' % ("latest"))
        torch.save(model.module.netD.state_dict(), path+'/%s_D.pth' % ("latest"))
        if not no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%s_EMA.pth' % ("latest"))
        with open(os.path.join(checkpoints_dir, name)+"/latest_iter.txt", "w") as f:
            f.write(str(cur_iter))
    elif best:
        torch.save(model.module.netG.state_dict(), path+'/%s_G.pth' % ("best"))
        torch.save(model.module.netD.state_dict(), path+'/%s_D.pth' % ("best"))
        if not no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%s_EMA.pth' % ("best"))
        with open(os.path.join(checkpoints_dir, name)+"/best_iter.txt", "w") as f:
            f.write(str(cur_iter))
    else:
        torch.save(model.module.netG.state_dict(), path+'/%d_G.pth' % (cur_iter))
        torch.save(model.module.netD.state_dict(), path+'/%d_D.pth' % (cur_iter))
        if not no_EMA:
            torch.save(model.module.netEMA.state_dict(), path+'/%d_EMA.pth' % (cur_iter))
