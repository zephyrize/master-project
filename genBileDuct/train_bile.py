from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data
from Vgg import Vgg16
from gycutils.trainschedule import Scheduler
from gycutils.utils import make_trainable,get_tv_loss,get_content_loss,get_content_features,get_style_features,get_style_loss, mkdir
from gan import Discriminator,Generator
from datasets import VOCDataSet
from torch.optim import Adam
from loss import BCE_Loss
from transform import ReLabel, ToLabel
from torchvision.transforms import Compose, Normalize, ToTensor,Resize
import tqdm
# from Criterion import Criterion
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os
from BileDuct_dataset_v4 import BileDuctDataset, RandomDataset
from torch.utils.data import DataLoader
from rich.progress import track
import torchvision.utils as vutils
import torchvision.transforms as transforms


save_folder = 'saved'
try:
    os.makedirs(save_folder)
except OSError:
    pass

sample_slices = 1
batch_size=24
# train_set = RandomDataset()
# validate_set = RandomDataset()
train_set = BileDuctDataset(split='train', 
                            sample_slices=sample_slices,
                            )
# validate_set = BileDuctDataset(split='val', sample_slices=sample_slices)

train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=batch_size*torch.cuda.device_count(), shuffle=True, drop_last=True)
# validate_loader = DataLoader(dataset=validate_set, num_workers=8, batch_size=1, shuffle=False, drop_last=False)

#########################################
#Parameters
#adversarial
L_gan_weight=1
#style
L_style_weight=10
#content
L_content_weight=1
#tv
L_tv_weight=100

lr=0.0002
beta1=0.5

max_epoch=500

channel=1
img_size=256
img_x=256
img_y=256

z_size=400
#########################################
def gen_rand_noise(batch_size, z_size, mean=0, std=0.001):
    z_sample = np.random.normal(mean, std, size=[batch_size, z_size]).astype(np.float32)
    z=torch.from_numpy(z_sample)
    return z
#########################################
G=Generator()
D=Discriminator()
Vgg=Vgg16()
# if torch.cuda.device_count() > 1:
#     G = torch.nn.DataParallel(G)
#     D = torch.nn.DataParallel(D)
#     Vgg = torch.nn.DataParallel(Vgg)
G.to('cuda')
D.to('cuda')
Vgg.to('cuda')
bce=BCE_Loss()
mse=torch.nn.MSELoss()
optimizer_d = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.9))
optimizer_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.9))
#########################################
#########################################


# fix_noise = torch.randn([batch_size, z_size]).cuda()

for epoch in tqdm.tqdm(range(max_epoch)):

    sample_num = 0
    mkdir('%s/epoch_%d'%(save_folder, epoch))

    for idx, sample in track(enumerate(train_loader), total=len(train_loader), description='train'):
        # trainD
        make_trainable(D, True)
        make_trainable(G, False)
        optimizer_d.zero_grad()

        real_img = sample['image'].cuda()
        real_label = sample['label'].unsqueeze(1).cuda()
        real_label_onehot = torch.nn.functional.one_hot(real_label.squeeze().long(), num_classes=2).permute(0, 3, 1, 2).float()  # 转换为one-hot编码
        z = torch.randn([batch_size, z_size]).cuda()

        fake_imgs = G(real_label_onehot, z)

        real_pair = torch.cat((real_img, real_label_onehot), dim=1)
        fake_pair = torch.cat((fake_imgs, real_label_onehot), dim=1)

        D_real_logits = D(real_pair)
        D_real_y = torch.ones((batch_size, 1)).cuda()

        D_fake_logits = D(fake_pair)
        D_fake_y = torch.zeros((batch_size,1)).cuda()

        d_real_loss = bce(D_real_logits, D_real_y)
        d_fake_loss = bce(D_fake_logits, D_fake_y)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        optimizer_d.step()


        # G.zero_grad()
        # optimizer_g.zero_grad()

        # z = torch.randn([batch_size, z_size]).cuda()
        # fake_imgs = G(real_label_onehot, z)
        # fake_pair = torch.cat((fake_imgs, real_label_onehot), dim=1)

        # D_fake_logits = D(fake_pair)
        # D_fake_y = torch.ones((batch_size, 1)).cuda()
        # #gan_loss
        # g_loss_adversial = bce(D_fake_logits, D_fake_y)
        # #tv_loss
        # # tv_loss = get_tv_loss(fake_imgs)
        # loss = L_gan_weight * g_loss_adversial # + L_tv_weight * tv_loss
        # loss.backward()
        # optimizer_g.step()
        #trainG twice
        make_trainable(G, True)
        make_trainable(D, False)
        make_trainable(Vgg,False)
        for _ in range(1):
            G.zero_grad()
            optimizer_g.zero_grad()

            # z = Variable(gen_rand_noise(batch_size, z_size, 0, 0.5)).cuda()
            z = torch.randn([batch_size, z_size]).cuda()
            fake_imgs = G(real_label_onehot, z)
            fake_pair = torch.cat((fake_imgs, real_label_onehot), dim=1)

            D_fake_logits = D(fake_pair)
            D_fake_y = Variable(torch.ones((batch_size, 1))).cuda()
            #gan_loss
            g_loss_adversial = bce(D_fake_logits, D_fake_y)
            #tv_loss
            # tv_loss = get_tv_loss(fake_imgs)
            loss = L_gan_weight * g_loss_adversial # + L_tv_weight * tv_loss

            style_feature = get_style_features(Vgg, real_img)
            style_loss = get_style_loss(style_feature, get_style_features(Vgg, fake_imgs))#/len(style_imgs)
            #content_loss
            content_loss=get_content_loss(get_content_features(Vgg, real_img),get_content_features(Vgg,fake_imgs))
            #tv_loss
            tv_loss=get_tv_loss(fake_imgs)
            loss=L_gan_weight*g_loss_adversial+L_style_weight*style_loss+L_content_weight*content_loss+L_tv_weight*tv_loss

            loss.backward()
            optimizer_g.step()

        if epoch >= 100 and (idx*batch_size) % 400 == 0:
            vutils.save_image(real_img.detach(), '%s/epoch_%d/real_image_%03d.png' % (save_folder, epoch, sample_num), normalize=True)
            vutils.save_image(real_label.detach(), '%s/epoch_%d/real_label_%03d.png' % (save_folder, epoch, sample_num), normalize=True)
            # 每个样本生成两份fake images
            # for i in range(2):
            #     z = torch.randn([batch_size, z_size]).cuda()
            #     fake = G(real_label, z)
            #     vutils.save_image(fake.detach(), '%s/epoch_%d/fake_samples_%03d_idx_%d.png' % (save_folder, epoch, sample_num, i+1), normalize=True)
            # sample_num = sample_num+1
            z = torch.randn([batch_size, z_size]).cuda()
            with torch.no_grad():
                fake = G(real_label_onehot, z)
                vutils.save_image(fake.detach(), '%s/epoch_%d/fake_samples_%03d.png' % (save_folder, epoch, sample_num), normalize=True)
            sample_num = sample_num+1
    print("epoch[%d/%d] d_loss:%.4f g_loss_ad:%.4f"%(epoch, max_epoch, d_loss, g_loss_adversial))

    if epoch >= 200:
        torch.save(G.state_dict(), '%s/epoch_%d/netG_epoch_%d.pth' % (save_folder, epoch, epoch))
        torch.save(D.state_dict(), '%s/epoch_%d/netD_epoch_%d.pth' % (save_folder, epoch, epoch))

    # if epoch%1==0:
    #     G.eval()
    #     D.eval()

    #     os.mkdir('%s/epoch_%d'%(save_folder, epoch))
    #     for idx, sample in track(enumerate(validate_loader), total=len(validate_loader), description='validate'):
    #         os.mkdir('%s/epoch_%d/label_%d' %(save_folder, epoch, idx) )

    #         real_img = Variable(sample['image']).cuda()
    #         real_label = Variable(sample['label'].unsqueeze(1)).cuda()
    #         img_label = real_label.squeeze().cpu().data[0].numpy()
    #         print(img_label.shape)
    #         Image.fromarray(img_label.astype(np.uint8)).save('%s/epoch_%d/label_%d/label.jpg'%(save_folder, epoch, idx))

    #         for i in range(5):
    #             z = Variable(gen_rand_noise(1, z_size, 0, 0.5)).cuda()
    #             fake_imgs = G(real_label, z)[0].cpu().data.numpy()
    #             img = np.transpose(fake_imgs,[1,2,0])

    #             img = (img+1)*127.5
    #             img = Image.fromarray(img.astype(np.uint8))
    #             img.save('%s/epoch_%d/label_%d/%d.jpg'%(save_folder, epoch, idx, i))

    


# CUDA_VISIBLE_DEVICES=7 python -W ignore train_bile.py