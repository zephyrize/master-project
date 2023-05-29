import torch
import os
from BileDuct_dataset_v4 import BileDuctDataset, RandomDataset
from torch.utils.data import DataLoader
import models.models as models
import models.losses as losses
import utils.utils as utils
from gycutils.utils import mkdir
import torchvision.utils as vutils
from rich.progress import track
import tqdm
from utils.fid_scores import fid_pytorch

save_folder = 'saved_oasis'
try:
    os.makedirs(save_folder)
except OSError:
    pass


sample_slices = 3
batch_size=6
max_epoch = 400
freq_save_ckpt = 400
freq_save_latest = 500
freq_fid = 400
freq_print = 200

train_set = BileDuctDataset(split='train', 
                            sample_slices=sample_slices,
                            )
# train_set = RandomDataset()

val_set = BileDuctDataset(split='val', 
                            sample_slices=sample_slices,
                            )
# val_set = RandomDataset()

train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=batch_size*torch.cuda.device_count(), shuffle=True, drop_last=True)
val_loader = DataLoader(dataset=val_set, num_workers=8, batch_size=batch_size*torch.cuda.device_count(), shuffle=False, drop_last=False)

#--- create utils ---#
losses_computer = losses.losses_computer()
visualizer_losses = utils.losses_saver()
fid_computer = fid_pytorch(val_loader)
im_saver = utils.image_saver()

#--- create models ---#
model = models.OASIS_model(phase='train')
model = models.put_on_multi_gpus(model)


#--- create optimizers ---#
lr_g = 0.0001
lr_d = 0.0004
beta1 = 0.0
beta2 = 0.999 
optimizerG = torch.optim.Adam(model.module.netG.parameters(), lr=lr_g, betas=(beta1, beta2))
optimizerD = torch.optim.Adam(model.module.netD.parameters(), lr=lr_d, betas=(beta1, beta2))

#--- the training loop ---#

for epoch in tqdm.tqdm(range(max_epoch)):
    sample_num = 0
    mkdir('%s/epoch_%d'%(save_folder, epoch))
    for i, data_i in track(enumerate(train_loader), total=len(train_loader), description='train'):

        cur_iter = epoch*len(train_loader) + i
        image, label = models.preprocess_input(data_i)
        
        #--- generator update ---#
        model.module.netG.zero_grad()
        loss_G, losses_G_list = model(image, label, "losses_G", losses_computer)
        loss_G, losses_G_list = loss_G.mean(), [loss.mean() if loss is not None else None for loss in losses_G_list]
        loss_G.backward()
        optimizerG.step()

        #--- discriminator update ---#
        model.module.netD.zero_grad()
        loss_D, losses_D_list = model(image, label, "losses_D", losses_computer)
        loss_D, losses_D_list = loss_D.mean(), [loss.mean() if loss is not None else None for loss in losses_D_list]
        loss_D.backward()
        optimizerD.step()
        
        # if cur_iter % freq_print == 0:
            # im_saver.visualize_batch(model, image, label, cur_iter)
        # if cur_iter % freq_save_ckpt == 0:
        #     utils.save_networks(cur_iter, model)
        # if cur_iter % freq_save_latest == 0:
        #     utils.save_networks(cur_iter, model, latest=True)
        if cur_iter % freq_fid == 0 and cur_iter > 0:
            is_best = fid_computer.update(model, cur_iter)
            if is_best:
                utils.save_networks(cur_iter, model, best=True)
                im_saver.visualize_batch(model, image, label, cur_iter)
        visualizer_losses(cur_iter, losses_G_list+losses_D_list)

        if (i*batch_size) % 50 == 0:
            vutils.save_image(image.detach(), '%s/epoch_%d/real_image_%03d.png' % (save_folder, epoch, sample_num), normalize=True)
            vutils.save_image(data_i['label'].unsqueeze(1).detach(), '%s/epoch_%d/real_label_%03d.png' % (save_folder, epoch, sample_num), normalize=True)
            fake = model(None, label, 'generate', None)
            vutils.save_image(fake.detach(), '%s/epoch_%d/fake_samples_%03d.png' % (save_folder, epoch, sample_num), normalize=True)
            sample_num += 1


utils.save_networks(cur_iter, model)
utils.save_networks(cur_iter, model, latest=True)
is_best = fid_computer.update(model, cur_iter)
if is_best:
    utils.save_networks(cur_iter, model, best=True)



# CUDA_VISIBLE_DEVICES=7 python -W ignore train_oasis.py

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train_oasis.py