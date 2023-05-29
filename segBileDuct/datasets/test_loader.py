import os
import torch 
import sys
sys.path.append('../')
import numpy as np
from torch.utils.data import Dataset, DataLoader

from datasets.loader_utils import *

class test_loader(Dataset):
    def __init__(self, args, file_path):

        self.args = args
        self.dataset_path = os.path.join(args.data_root_dir, args.dataset, 'preprocess')
        self.voxels_all = None
        self.file_path = file_path
        self.re_size = 256
        self.seg_type = args.seg_type

        # calculate the label voxels in the trainset
        if self.voxels_all is None:
            self.voxels_all = get_voxels_all(load_file_name_list(os.path.join(self.dataset_path, 'train_path_info.txt'))) # here must ensure that voxels is from train set
            self.voxels_mean = np.mean(self.voxels_all)
            self.voxels_std = np.std(self.voxels_all)

        self.stack_ct, self.stack_label = self._get_image_list()



    def __getitem__(self, index):

        ct = self.stack_ct[index]
        label = self.stack_label[index]

        ct = torch.FloatTensor(ct)
        label = torch.FloatTensor(label)

        if self.seg_type == '2D':
            ct = ct.unsqueeze(0)
            label = label.unsqueeze(0)

        if self.args.loss_func == 'lognll':
            label = label.squeeze()
        
        return ct, label 

    def __len__(self):
        return self.stack_ct.shape[0]


    def _get_image_list(self):
        
        ct_list = []
        label_list = []

        ct, label = preprocess(self, self.file_path)

        ct_list.append(ct)
        label_list.append(label)

        
        stack_ct = ct_list[0]
        stack_label = label_list[0]

        for i in range(1, len(ct_list)):
            stack_ct = np.vstack([stack_ct, ct_list[i]])
            stack_label = np.vstack([stack_label, label_list[i]])

        return stack_ct, stack_label


if __name__ == '__main__':


    from config import args
    
    filename = 'test_path_info.txt'
    
    filename_list = load_file_name_list(os.path.join(args.data_root_dir, args.dataset, 'preprocess', filename))
    data_set = test_loader(args, filename_list[0])
    print('len: ', len(data_set))

    data_loader = DataLoader(data_set, 16, False, num_workers=8)

    for i, (ct, seg) in enumerate(data_loader):
        print("ct: {},   seg: {}".format(ct.shape, seg.shape))









