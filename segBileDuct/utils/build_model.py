from ast import arg
from fileinput import filename
import os
from os.path import join
from networks.Unet import *
from networks.UnetASPP import *
from networks.MNet import *
from networks.swinUnet.vision_transformer import get_swin_unet
from networks.transUnet import *
from networks.official_transUnet.get_transUnet import get_transUnet
from networks.axialnet import axialunet, gated, MedT, logo
from networks.SATr import get_Unet_SATR
from networks.AATM import get_Unet_AATM
from networks.AATM_V2 import get_Unet_AATM_V2
from networks.AATM_V3 import get_Unet_AATM_V3
from networks.AATM_V4 import get_Unet_AATM_V4
from networks.AATM_V5 import get_Unet_AATM_V5
from networks.AATM_V6 import get_Unet_AATM_V6
from networks.AATM_V7 import get_Unet_AATM_V7
from networks.AATM_V8 import get_Unet_AATM_V8
from networks.AATM_V10 import get_Unet_AATM_V10
from networks.AATM_V12 import get_Unet_AATM_V12
from networks.CAT_Net import get_cat_net

from networks.ATM import get_Unet_ATM
from networks.ATM_V1 import get_Unet_ATM_V1
from networks.ATM_V2 import get_Unet_ATM_V2
from networks.ATM_V3 import get_Unet_ATM_V3
from networks.ATM_V5 import get_Unet_ATM_V5
from networks.ATM_V6 import get_Unet_ATM_V6
from networks.ATM_V7 import get_Unet_ATM_V7
from networks.ATM_V8 import get_Unet_ATM_V8
from networks.ATM_V9 import get_Unet_ATM_V9

from networks.PDC_ATM import get_Unet_PDC_ATM
from networks.PDCNet import get_Unet_PDCNet

from networks.SATNet import get_SATNet

from networks.CorAtt import *
from networks.AttentionUnet import *
from networks.NestedUnet import *
from networks.axialAttentionUnet import *
from networks.MT_Unet import *

from networks.TAM import *

import utils.losses as losses
from utils.helper import load_file_name_list
from torch.utils.data import DataLoader

from datasets.data_loader import data_loader
from datasets.test_loader import test_loader

from monai.losses import DiceLoss
from monai.losses import GeneralizedDiceLoss
from monai.losses import DiceCELoss
from monai.losses import DiceFocalLoss
from monai.losses import FocalLoss
from monai.losses import TverskyLoss
import torch.nn.functional as F

def get_criterion(args, device=None):

    if args.loss_func == 'dice':
        criterion = DiceLoss(include_background=False).to(device)
    if args.loss_func == 'sat':
        criterion = losses.SATLoss().to(device)
    if args.loss_func == 'generalDice':
        criterion = GeneralizedDiceLoss(include_background=False).to(device)
    elif args.loss_func == 'diceCE':
        criterion = DiceCELoss(include_background=False).to(device)
    elif args.loss_func == 'diceFocal':
        criterion = DiceFocalLoss(include_background=False).to(device)
    elif args.loss_func == 'lognll':
        criterion = losses.LogNLLLoss().to(device)
    elif args.loss_func == 'focal':
        criterion = FocalLoss(include_background=False).to(device)
    elif args.loss_func == 'tversky':
        criterion = TverskyLoss(include_background=False).to(device)
    elif args.loss_func == 'ftloss':
        criterion = losses.FTLOSS().to(device)
    elif args.loss_func == 'pdc':
        criterion = losses.PDCLoss().to(device)
    
    return criterion

def get_data_loader(args):

    file_name = 'split_train_val.json'
    train_set = data_loader(args, file_name, mode='train', process_type=args.process_type, sample_slices=args.sample_slices)
    val_set = data_loader(args, file_name, mode='val', process_type=args.process_type, sample_slices=args.sample_slices)
    
    print('length of batch sampler: ', len(train_set))
    
    train_load = DataLoader(dataset=train_set, batch_size=args.batch_size * len(args.gpu), shuffle=True, num_workers=8, drop_last=True)
    val_load = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=8)
    # logging.info('\nload data done...')
    return train_load, val_load


def get_model(args, mode='train', device=None, device_ids=None):
    
    img_ch = args.sample_slices

    if args.model == 'SATNet':
        model =  get_SATNet(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)

    if args.model == 'Cat':
        model = get_cat_net(img_ch, batch_size=args.batch_size * len(args.gpu))

    if args.model == 'Unet':
        model = Unet(img_ch, 1)

    if args.model == 'UnetASPP':
        model = UnetASPP(img_ch, 1)
    
    if args.model == 'MNet':
        model = MNet(img_ch, 1)

    if args.model == 'CorAtt':
        model = CorAtt_Unet(img_ch, 1)
    
    if args.model == 'TAM':
        model = TAM_Unet(1, 1)

    if args.model == 'transUnet':
        model = TransUNet(img_dim=256,
                          in_channels=img_ch,
                          out_channels=128,
                          head_num=4,
                          mlp_dim=512,
                          block_num=8,
                          patch_dim=16,
                          class_num=1)
    
    if args.model == 'offical_transUnet':
        model = get_transUnet(img_size=256, mode=mode)
    
    if args.model == 'PDCNet':
        model = get_Unet_PDC_ATM(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)
        
    if args.model == 'pdcnet':
        model =  get_Unet_PDCNet(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)


    if args.model == 'ATM':
        model = get_Unet_ATM(img_size=256, dsv=args.dsv)

    if args.model == 'ATM_V1':
        model = get_Unet_ATM_V1(img_size=256, dsv=args.dsv)

    if args.model == 'ATM_V2':
        model = get_Unet_ATM_V2(img_size=256, dsv=args.dsv)

    if args.model == 'ATM_V3':
        model = get_Unet_ATM_V3(img_size=256, dsv=args.dsv)
    
    if args.model == 'ATM_V5':
        model = get_Unet_ATM_V5(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)

    if args.model == 'ATM_V6':
        model = get_Unet_ATM_V6(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)

    if args.model == 'ATM_V7':
        model = get_Unet_ATM_V7(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)

    if args.model == 'ATM_V8':
        model = get_Unet_ATM_V8(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)

    if args.model == 'ATM_V9':
        model = get_Unet_ATM_V9(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)

    if args.model == 'Unet_SATR':
        model = get_Unet_SATR(img_size=256, dsv=args.dsv)

    if args.model == 'AATM':
        model = get_Unet_AATM(img_size=256, dsv=args.dsv)
    
    if args.model == 'AATM_V2':
        model = get_Unet_AATM_V2(img_size=256)

    if args.model == 'AATM_V3':
        model = get_Unet_AATM_V3(img_size=256)

    if args.model == 'AATM_V4':
        model = get_Unet_AATM_V4(img_size=256, dsv=args.dsv)

    # if args.model == 'AATM_V11':
    #     model = get_Unet_AATM_V11(img_size=256, dsv=args.dsv)
    
    if args.model == 'AATM_V12':
        model = get_Unet_AATM_V12(img_size=256, dsv=args.dsv)

    if args.model == 'AATM_V5':
        model = get_Unet_AATM_V5(img_size=256, dsv=args.dsv)

    if args.model == 'AATM_V6':
        model = get_Unet_AATM_V6(img_size=256, dsv=args.dsv, scale = args.scale)
    
    if args.model == 'AATM_V7':
        model = get_Unet_AATM_V7(img_size=256, dsv=args.dsv)

    if args.model == 'AATM_V8':
        model = get_Unet_AATM_V8(img_size=256, dsv=args.dsv)
    
    if args.model == 'AATM_V10':
        model = get_Unet_AATM_V10(img_size=256, dsv=args.dsv)

    if args.model == 'swin_unet':
        model = get_swin_unet(img_size=256, out_ch=1)

    if args.model == 'axialAttentionUnet':
        model = get_axial_attention_model(args)

    if args.model == 'axialUnet':
        
        num_classes = None
        if args.loss_func == 'lognll':
            num_classes = 2
        else:
            num_classes = 1
        model = axialunet(img_size=256, s=0.125, imgchan=img_ch, num_classes=num_classes)

    if args.model == 'gated':
        model = gated(img_size=256, s=args.gated_scale, imgchan=img_ch)

    if args.model == 'medt_net':
        model = MedT(img_size=256, imgchan=img_ch)
    
    if args.model == 'logo':
        model = logo(img_size=256, imgchan=img_ch)

    if args.model == 'AttUnet':
        model = AttUnet(img_ch, 1)
    
    if args.model == 'NestedUnet':
        model = NestedUnet(img_ch, 1)

    if args.model == 'MTUNet':
        model = MTUNet(out_ch=1)


    if len(device_ids) > 1:
        print('model -> nn.DataParallel')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
    else:
        model = model.to(device)

    # if args.mulGPU is False:
    #     model = model.to(device)
    # else:
    #     model.cuda()
    #     model = torch.nn.DataParallel(model, device_ids=[0])
    

    if args.continue_training is True:
        if args.model_remark != '':
            args.model_remark = '_' + args.model_remark
            model_path = join('model', args.dataset, args.model + args.model_remark + '_finalModel.pt')
            model.load_state_dict(torch.load(model_path, map_location=device))
        return model
        

    if mode == 'test':
        if args.model_remark != '':
            args.model_remark = '_' + args.model_remark

        if args.best_model is True:
            model_path = join('model', args.dataset, args.model + args.model_remark + '_withBestModel.pt')
        elif args.final_model is True:
            model_path = join('model', args.dataset, args.model + args.model_remark + '_finalModel.pt')
        else:
            model_path = join('model', args.dataset, args.model + args.model_remark + '_checkpoint.pt')

        if args.use_cpu is True:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # 'cuda:0'
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        print("model path: ", model_path)
    
    return model