import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'
import numpy as np
import random
import torch
import argparse
from os.path import join, exists
# import gc
import sys
from tqdm import tqdm
sys.path.append('../')

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable


from medseg.common_utils.load_args import Params
from medseg.common_utils.metrics import print_metric
from medseg.common_utils.basic_operations import check_dir, link_dir, delete_dir, set_seed
from medseg.dataset_loader.BileDuct_dataset_v3 import BileDuctDataset
from medseg.models.model_util import makeVariable
from medseg.models.advanced_triplet_recon_segmentation_model import AdvancedTripletReconSegmentationModel
from medseg.models.advanced_triplet_recon_segmentation_model_2 import AdvancedTripletReconSegmentationModel_2
from medseg.models.advanced_triplet_recon_segmentation_model_3 import AdvancedTripletReconSegmentationModel_3
from medseg.dataset_loader.transform_bile import Transformations


class OpenDataParallel(torch.nn.DataParallel):
    """custom class so that I can have other attributes than modules."""

    def __init__(self, module, **kwargs):
        super(OpenDataParallel, self).__init__(module, **kwargs)

    def __getattr__(self, name):
        if name != 'module':
            try:
                return getattr(self.module, name)
            except AttributeError:
                pass

        return super(torch.nn.DataParallel, self).__getattr__(name)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def sample_batch(dataiter, dataloader, SSL_flag=False):
    try:
        batch = next(dataiter)
    except StopIteration:
        dataiter = dataloader.__iter__()
        batch = next(dataiter)

    if SSL_flag: # semi supervised learning
        labelled_batch, unlabelled_batch = batch
    else:
        labelled_batch = batch
        unlabelled_batch = None
    return labelled_batch, unlabelled_batch, dataiter


def get_batch(dataiter, train_loader, use_gpu=True, keep_origin=True, mode='train'):

    labelled_batch, unlabelled_batch, dataiter = sample_batch(
        dataiter=dataiter, dataloader=train_loader, SSL_flag=False)
    image_l, label_l = labelled_batch['image'], labelled_batch['label']
    
    if keep_origin:
        image_orig, gt_orig = labelled_batch['origin_image'], labelled_batch['origin_label']
        image_l = torch.cat([image_l, image_orig], dim=0)
        label_l = torch.cat([label_l, gt_orig], dim=0)
    image_l = makeVariable(image_l, type='float',
                           use_gpu=use_gpu, requires_grad=False)
    label_l = makeVariable(label_l, type='long',
                           use_gpu=use_gpu, requires_grad=False)
    # print(image_l.shape, label_l.shape)

    return image_l, label_l, unlabelled_batch, dataiter, int(labelled_batch['pid']) if mode != 'train' else None

def eval_model(segmentation_model, validate_loader, val_dataiter, keep_origin=False):
    segmentation_model.eval()
    segmentation_model.running_metric.reset()

    for b_iter in range(len(validate_loader)):
        clean_image_l, label_l, unlabelled_batch, val_dataiter, pid = get_batch(
            val_dataiter, validate_loader, keep_origin=keep_origin, use_gpu=use_gpu, mode='val')
        random_sax_image_V = makeVariable(clean_image_l, type='float',
                                          use_gpu=use_gpu, requires_grad=False)
        # use STN's performance for model selection
        segmentation_model.evaluate(input=random_sax_image_V,
                                    targets_npy=label_l.cpu().data.numpy(), n_iter=2, pid=pid)
    # score = print_eval_metric(segmentation_model.running_metric,
    #                      name=experiment_name)
    # curr_score = score['Mean IoU : \t']

    segmentation_model.running_metric.print_metric()

    eval_dice = segmentation_model.running_metric.get_eval_dice()
    print('AVG Dice: ', eval_dice)

    return eval_dice


def train_network(experiment_name, dataset,
                  segmentation_solver,
                  experiment_opt,
                  log=False,
                  debug=False, use_gpu=True):
    '''

    :param experiment_name:
    :param dataset:

    :param resume_path:
    :param log:
    :return:
    '''
    # output setting
    global start_epoch, last_epoch, training_opt, crop_size, model_dir, log_dir

    # =========================dataset config==================================================#
    train_set = dataset[0]
    validate_set = dataset[1]
    batch_size = experiment_opt['learning']["batch_size"]
    assert batch_size >= 1, f'batch size must >=1, but got {batch_size}'
    if experiment_opt['data']['keep_orig_image_label_pair_for_training']:
        train_batch_size = batch_size//2
        if train_batch_size == 0:
            train_batch_size = 1
    else:
        train_batch_size = batch_size

    g = torch.Generator()
    if training_opt.seed is not None:
        g.manual_seed(training_opt.seed)
    train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=int(train_batch_size)*torch.cuda.device_count(), shuffle=True, drop_last=True, worker_init_fn=seed_worker, generator=g)
    validate_loader = DataLoader(dataset=validate_set, num_workers=8, batch_size=1, shuffle=False, drop_last=False, worker_init_fn=seed_worker, generator=g)

    best_score = -10000

    if log:
        writer = SummaryWriter(log_dir=log_dir, purge_step=start_epoch)

   ##########latent DA and cooperative training config############

    latent_DA = experiment_opt['learning']['latent_DA']
    separate_training = experiment_opt['learning']['separate_training']

    latentDA_config = experiment_opt["latent_DA"]
    gen_corrupted_image = False
    gen_corrupted_seg = False
    corrupted_seg_DA_config = None
    corrupted_image_DA_config = None
    if latent_DA:
        print('latent code masking configurations')
        if 'image code' in latentDA_config["mask_scope"]:
            gen_corrupted_image = True
            corrupted_image_DA_config = latentDA_config["image code"]
            print(latentDA_config["image code"])
        if 'shape code' in latentDA_config["mask_scope"]:
            gen_corrupted_seg = True
            print(latentDA_config["shape code"])
            corrupted_seg_DA_config = latentDA_config["shape code"]

    # =========================<<<<<start training>>>>>>>>=============================>
    i_iter = 0
    stop_flag = False
    score_list = []
    dataiter = iter(train_loader)
    val_dataiter = iter(validate_loader)

    keep_origin = experiment_opt['data']['keep_orig_image_label_pair_for_training']

    segmentation_solver.reset_all_optimizers()
    segmentation_solver.train()
    device = torch.device('cuda') if use_gpu else torch.device('cpu')

    try:
        for i_epoch in range(start_epoch, experiment_opt['learning']['n_epochs']):
            last_epoch = i_epoch
            # torch.cuda.empty_cache()
            # gc.collect()  # collect garbage
            g_count = 0
            total_loss = 0.
            loss_keys = ['loss/standard/total', 'loss/standard/seg', 'loss/standard/image', 'loss/standard/shape', 'loss/standard/gt_shape',
                         'loss/hard/total', 'loss/hard/seg', 'loss/hard/image', 'loss/hard/shape'
                         ]
            loss_dict = {}
            for key in loss_keys:
                loss_dict[key] = torch.tensor(0., device=device)

            print("begin epoch {} training...".format(i_epoch))
            for i_iter in tqdm(range(len(train_loader))):
                if stop_flag:
                    break
                # gc.collect()  # collect garbage
                # step 1: get a batch of labelled images to get initial estimate
                segmentation_solver.train()
                segmentation_solver.reset_all_optimizers()

                clean_image_l, label_l, unlabelled_batch, dataiter, _ = get_batch(
                    dataiter, train_loader, keep_origin=keep_origin, use_gpu=use_gpu, mode='train')
                clean_image_l = makeVariable(
                    clean_image_l, use_gpu=use_gpu, requires_grad=False, type='float')
                batch_4d_size = clean_image_l.size()
                # add noise to input to train the FTN (same as training a denoising autoencoder)
                noise = 0.05 * torch.randn(batch_4d_size[0], batch_4d_size[1], batch_4d_size[2],
                                           batch_4d_size[3], device=clean_image_l.device, dtype=clean_image_l.dtype)
                image_l = torch.clamp(clean_image_l + noise, 0, 1)
                image_l = makeVariable(image_l.detach().clone(), use_gpu=use_gpu,
                                       requires_grad=False, type='float')
                # step 2: standard training
                # print("begin step2......")
                seg_loss, image_recon_loss, gt_recon_loss, shape_recon_loss = segmentation_solver.standard_training(
                    clean_image_l, label_l, perturbed_image=image_l, separate_training=separate_training)
                
                # print("standard training: seg_loss: {}, image_recon_loss: {}, gt_recon_loss: {}, shape_recon_loss: {}".format(
                #     seg_loss, image_recon_loss, gt_recon_loss, shape_recon_loss
                # ))
                # print("end step2......")
                standard_loss = seg_loss + image_recon_loss + shape_recon_loss+gt_recon_loss
                loss_dict['loss/standard/total'] += standard_loss.item()
                loss_dict['loss/standard/seg'] += seg_loss.item()
                loss_dict['loss/standard/image'] += image_recon_loss.item()
                loss_dict['loss/standard/shape'] += shape_recon_loss.item()
                loss_dict['loss/standard/gt_shape'] += gt_recon_loss.item()

                if latent_DA:
                    segmentation_solver.reset_all_optimizers()
                    perturbed_image_0, perturbed_y_0 = segmentation_solver.hard_example_generation(clean_image_l.detach().clone(),
                                                                                                   label_l.detach().clone(),
                                                                                                   gen_corrupted_seg=gen_corrupted_seg,
                                                                                                   gen_corrupted_image=gen_corrupted_image,
                                                                                                   corrupted_image_DA_config=corrupted_image_DA_config,
                                                                                                   corrupted_seg_DA_config=corrupted_seg_DA_config
                                                                                                   )

                    seg_supervised_loss, corrupted_image_recon_loss, shape_recon_loss_2, corrupted_shape_recon_loss = segmentation_solver.hard_example_training(perturbed_image=perturbed_image_0,
                                                                                                                                                                perturbed_seg=perturbed_y_0,
                                                                                                                                                                clean_image_l=clean_image_l, label_l=label_l,
                                                                                                                                                                separate_training=separate_training)
                    # print("latent_DA: seg_supervised_loss: {}, corrupted_image_recon_loss: {}, shape_recon_loss_2: {}, corrupted_shape_recon_loss: {}".format(
                    #     seg_supervised_loss, corrupted_image_recon_loss, shape_recon_loss_2, corrupted_shape_recon_loss
                    # ))  

                    hard_loss = seg_supervised_loss + corrupted_image_recon_loss + \
                        shape_recon_loss_2 + corrupted_shape_recon_loss
                    loss_dict['loss/hard/total'] += hard_loss.item()
                    loss_dict['loss/hard/seg'] += seg_supervised_loss.item()
                    loss_dict['loss/hard/image'] += corrupted_image_recon_loss.item()
                    loss_dict['loss/hard/shape'] += (shape_recon_loss_2 +
                                                     corrupted_shape_recon_loss).item()
                    # torch.cuda.empty_cache()

                else:
                    hard_loss = torch.tensor(0., device=device)

                loss = standard_loss + hard_loss
                segmentation_solver.reset_all_optimizers()
                loss.backward()
                segmentation_solver.optimize_all_params()
                total_loss += loss.item()
                # torch.cuda.empty_cache()
                g_count += 1
                i_iter += 1
                if i_iter > experiment_opt['learning']["max_iteration"]:
                    stop_flag = True

            print('{} network: {} epoch {} training loss iter: {}, total  loss: {}'.
                  format(experiment_name, experiment_opt['segmentation_model']["network_type"], i_epoch, g_count, str(total_loss / (1.0 * g_count))))
            if log:
                if debug:
                    print('logging w. tensorboard')
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(
                        loss_name, (loss_value / (1.0 * g_count)), i_epoch)

            # =========================<<<<<start evaluating>>>>>>>>=============================>
            curr_score = eval_model(
                segmentation_solver, validate_loader, val_dataiter, keep_origin=False)
            score_list.append(curr_score)

            if log:
                writer.add_scalar('dice/val_dice', curr_score, i_epoch)
                # writer.add_scalar('acc/val_acc', curr_acc, i_epoch)
            # save best models
            if best_score < curr_score:
                best_score = curr_score
                segmentation_solver.save_model(model_dir, epoch_iter='best',
                                               model_prefix=experiment_opt['segmentation_model']["network_type"])
                segmentation_solver.save_testing_images_results(
                    model_dir, epoch_iter='best', max_slices=5)

            ###########save outputs ####################################################################
            if (i_epoch + 1) % experiment_opt["output"]["save_epoch_every_num_epochs"] == 0 or i_epoch == 0:
                segmentation_solver.save_model(model_dir, epoch_iter=i_epoch,
                                               model_prefix=experiment_opt['segmentation_model']["network_type"])
                segmentation_solver.save_testing_images_results(
                    model_dir, epoch_iter=i_epoch, max_slices=5)
                # gc.collect()  # collect garbage

            if stop_flag:
                break
            # avoid pytorch vram(GPU memory) usage keeps increasing
            # torch.cuda.empty_cache()

        if log:
            try:
                writer.export_scalars_to_json(
                    join(log_dir, experiment_name + ".json"))
                writer.close()
            except:
                print('already closed')
    except Exception as e:
        print('catch exception at epoch {}. error: {}'.format(str(i_epoch), e))
        if i_epoch > 0:
            segmentation_solver.save_snapshots(model_dir, epoch=i_epoch)
        last_epoch = i_epoch


# ========================= config==================================================#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='cooperative training and latent space DA for robust segmentation')
    # training config setting
    parser.add_argument("--json_config_path", type=str, default='../config/BileDuct/standard_training.json',
                        help='path of configurations')
    # data setting
    parser.add_argument("--dataset_name", type=str, default='BileDuct',
                        help='dataset name')
    parser.add_argument("--cval", type=int, default=0,
                        help="cross validation subset")
    parser.add_argument("--data_setting", type=str, default="10",
                        help="data_setting:['one_shot','three_shot']")

    # output setting
    parser.add_argument("--resume_pkl_path", type=str,
                        default=None, help='path-to-model-snapshot.pkl')
    parser.add_argument("--save_dir", type=str,
                        default="./saved/",
                        help='path to resume the models')
    # visualizing the training/test performance
    parser.add_argument("--log", action='store_true', default=False,
                        help='use tensorboardX to tracking the training and testing curve')
    # advanced setting
    parser.add_argument("--seed", type=int, default=None,
                        help="set seed to reduce randomness in training")
    parser.add_argument("--n_workers", type=int, default=8,
                        help="number of workers for data loaders")
    parser.add_argument("--no_pin_memory", action='store_true', default=False,
                        help='use pin memory for speed-up')
    parser.add_argument("--debug", action='store_true', default=False,
                        help='print information for debugging')
    

    # ========================= initialize training settings==================================================#
    # first load basic settings and then load args, finally load experiment configs
    training_opt = parser.parse_args()
    # limit randomness for reproducible research, ref: https://pytorch.org/docs/stable/notes/randomness.html
    # global setting

    set_seed(training_opt.seed)

    if training_opt.debug:
        import faulthandler
        faulthandler.enable()

    config_path = training_opt.json_config_path

    if exists(config_path):
        print('load params from {}'.format(config_path))
        experiment_opt = Params(config_path).dict
    else:  #
        print(config_path + ' does not not exist')
        raise FileNotFoundError
    
    # input dataset setting
    data_opt = experiment_opt['data']
    data_aug_policy_name = data_opt["data_aug_policy"]
    crop_size = data_opt['crop_size']

    # ========================= initialize training settings==================================================#
    print("begin loading data...")

    # trans = Transformations(
    #     data_aug_policy_name=data_aug_policy_name
    # ).get_transformation()

    # train_set = BileDuctDataset(
    #     root_dir=data_opt["root_dir"],
    #     split='train', transform=trans['train'],
    #     num_classes=data_opt["num_classes"],
    #     image_size=data_opt['image_size'],
    #     label_size=data_opt['label_size'],
    #     cval=training_opt.cval,
    #     keep_orig_image_label_pair=data_opt['keep_orig_image_label_pair_for_training'],
    #     sample_slices=1
    # )

    # validate_set = BileDuctDataset(
    #     root_dir=data_opt["root_dir"],
    #     split='val', transform=trans['val'],
    #     num_classes=data_opt["num_classes"],
    #     image_size=data_opt['image_size'],
    #     label_size=data_opt['label_size'],
    #     cval=training_opt.cval,
    #     keep_orig_image_label_pair=False,
    #     sample_slices=1
    # )

    if experiment_opt['segmentation_model']["network_type"] == 'FCN_16_standard':
        sample_slices = 1
    else:
        sample_slices = 3
    train_set = BileDuctDataset(split='train', sample_slices=sample_slices)
    validate_set = BileDuctDataset(split='val', sample_slices=sample_slices, tv=True)

    # train_set = RandomDataset()
    # validate_set = RandomDataset()

    print("loading data done...")

    datasets = [train_set, validate_set]

    # ========================Define models==================================================#
    num_classes = experiment_opt['segmentation_model']["num_classes"]
    network_type = experiment_opt['segmentation_model']["network_type"]
    start_epoch = 0
    learning_rate = experiment_opt['learning']['lr']
    use_gpu = experiment_opt['learning']["use_gpu"]

    segmentation_solver = AdvancedTripletReconSegmentationModel_3(network_type=network_type,
                                                                image_ch=sample_slices, num_classes=num_classes,
                                                                learning_rate=learning_rate,
                                                                use_gpu=use_gpu,
                                                                n_iter=1,
                                                                checkpoint_dir=None,
                                                                debug=training_opt.debug
                                                                )


    if training_opt.resume_pkl_path is not None:
        start_epoch = segmentation_solver.load_snapshots(
            training_opt.resume_pkl_path)
        print(f'training starts at {start_epoch}')
    last_epoch = start_epoch


    # ========================= start training ==================================================#
    project_str = 'train_{}_{}_n_cls_{}'.format(data_opt['dataset_name'], str(
        training_opt.data_setting), str(experiment_opt['segmentation_model']["num_classes"]))

    global_dir = training_opt.save_dir
    save_dir = join(training_opt.save_dir, project_str)
    config_name = training_opt.json_config_path.replace("../config/", "")
    config_name = config_name.replace(".json", "")
    experiment_name = "{exp}/{cval}".format(
        exp=config_name, cval=str(training_opt.cval))
    log_dir = join(global_dir, *[project_str, experiment_name, 'log'])
    model_dir = join(global_dir, *[project_str, experiment_name, 'model'])
    check_dir(log_dir, create=True)
    check_dir(model_dir, create=True)
    print(f'create {model_dir} to save trained models')
    # torch.cuda.empty_cache()

    try:
        train_network(experiment_name=experiment_name,
                      dataset=datasets,
                      segmentation_solver=segmentation_solver,
                      experiment_opt=experiment_opt,
                      log=training_opt.log,
                      debug=training_opt.debug)
    except KeyboardInterrupt:
        print('keyboardInterrupted')
        if last_epoch > 0:
            save_path = segmentation_solver.save_snapshots(
                model_dir, epoch=last_epoch)
            print(f'save snapshots at epoch {last_epoch} to {save_path}')


'''
CUDA_VISIBLE_DEVICES=6,7 python -W ignore train_bile.py --log --data_setting "keep-origin" --cval 2

'''


'''
train_BileDuct_10_n_cls_2/BileDut
standard_training
0 - FCN 
cooperative_training
0 - FCN -------- lr:1e-5 
CUDA_VISIBLE_DEVICES=6,7 python -W ignore train_bile.py --log --cval 0 --json_config_path ../config/BileDuct/cooperative_training.json
1 - FCN -------- lr:1e-4;batch:12
CUDA_VISIBLE_DEVICES=4,5 python -W ignore train_bile.py --log --cval 1 --json_config_path ../config/BileDuct/cooperative_training.json
2 - FCN -------- lr:5*1e-5;batch:16 keep-origin-true
CUDA_VISIBLE_DEVICES=6,7 python -W ignore train_bile.py --log --cval 2 --json_config_path ../config/BileDuct/cooperative_training.json
'''



'''
train_BileDuct_keep-origin-false_n_cls_2/BileDut
standard_training
0 - TAGNet -------- batch 6; lr 1e-5; 
CUDA_VISIBLE_DEVICES=5,6,7 python -W ignore train_bile.py --log --cval 0 --data_setting keep-origin-false

1 - TAGNet -------- batch 6; lr 1e-5;训练集和验证集混合训练
CUDA_VISIBLE_DEVICES=5,6,7 python -W ignore train_bile.py --log --cval 1 --data_setting keep-origin-false

2 - TAGNet -------- batch 6; lr 1e-5;验证集为测试集,去除混合训练
CUDA_VISIBLE_DEVICES=5,6,7 python -W ignore train_bile.py --log --cval 2 --data_setting keep-origin-false

3 - TAGNet -------- batch 6; lr 1e-5; image decoder 添加 skip connection 损失函数仍然是ce; 训练集和验证集混合训练
CUDA_VISIBLE_DEVICES=5,6,7 python -W ignore train_bile.py --log --cval 3 --data_setting keep-origin-false

4 - TAGNet -------- batch 6; lr 1e-5; 相对于0，更改重建分支损失为ssim loss
CUDA_VISIBLE_DEVICES=5,6,7 python -W ignore train_bile.py --log --cval 4 --data_setting keep-origin-false

cooperative_training

1 - TAGNet -------- batch 3; lr 5*1e-5;训练集和验证集混合训练
CUDA_VISIBLE_DEVICES=1,2,3,4 python -W ignore train_bile.py --log --cval 1 --data_setting keep-origin-false --json_config_path ../config/BileDuct/cooperative_training.json

3 - TAGNet -------- batch 3; lr 1e-4;  对标 standard_training -3
CUDA_VISIBLE_DEVICES=1,2,3,4 python -W ignore train_bile.py --log --cval 3 --data_setting keep-origin-false --json_config_path ../config/BileDuct/cooperative_training.json

4 - TAGNet -------- batch 3; lr 1e-4;  loss为 ssim loss，权重为0.8，window size 为16
CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore train_bile.py --log --cval 4 --data_setting keep-origin-false --json_config_path ../config/BileDuct/cooperative_training.json


5 - TAGNet -------- batch 3; lr 1e-5;  loss为 mse  权重0.7 草。模型文件夹不小心删除了，只剩log了
CUDA_VISIBLE_DEVICES=5,6,7 python -W ignore train_bile.py --log --cval 5 --data_setting keep-origin-false --json_config_path ../config/BileDuct/cooperative_training.json --seed 3407

6 - TAGNet -------- batch 3; lr 1e-5;  loss为 ssim  权重0.7
CUDA_VISIBLE_DEVICES=2,3,4 python -W ignore train_bile.py --log --cval 6 --data_setting keep-origin-false --json_config_path ../config/BileDuct/cooperative_training.json --seed 3407

'''

'''
train_BileDuct_keep-origin-true_n_cls_2/BileDut
0 - TAGNet -------- batch 6; lr 1e-4; 
CUDA_VISIBLE_DEVICES=2,3,4 python -W ignore train_bile.py --log --cval 0 --data_setting keep-origin-true
'''