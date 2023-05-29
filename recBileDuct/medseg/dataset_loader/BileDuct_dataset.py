import os
from os.path import join
import json
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch 
import h5py
# from medseg.dataset_loader.base_segmentation_dataset import BaseSegDataset
from medseg.dataset_loader.dataset_utils import normalize_minmax_data

class BileDuctDataset(Dataset):
    def __init__(self, filename="split_train_val.json", mode='train', process_type='2D', sample_slices=3):

        self.mode = mode
        self.type = process_type

        self.dataset_dir = join('/data1/zfx/data/', 'BileDuct')

        if self.mode == 'train':
            self.filename_list = load_json(join(self.dataset_dir, 'preprocessed_data', filename))['train']
        elif self.mode == 'val':
            self.filename_list = load_json(join(self.dataset_dir, 'preprocessed_data', filename))['val']
        elif self.mode == 'test':
            self.filename_list = load_json(join(self.dataset_dir, 'preprocessed_data', "preprocess_dataset.json"))['test']

        self.train_data_path = join(self.dataset_dir, 'preprocessed_data', 'train_data.h5')
        self.val_data_path = join(self.dataset_dir, 'preprocessed_data', 'val_data.h5')

        if self.mode == 'train' or self.mode == 'val':
            self.image, self.label = self._get_image_list(self.filename_list)
        else:
            self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict = self._scan_test_data(self.filename_list)
            self.patient_number = len(self.patient_id_list)
            self.slice_id = 0

            self.idx2cls_dict = {}
            for i in range(2):
                self.idx2cls_dict[i] = str(i)
            self.formalized_label_dict = self.idx2cls_dict

        self.extend_slice = sample_slices // 2

        self.pid_map = {
            '002' : 0,
            '010' : 1,
            '012' : 2
        }

        self.data_aug = iaa.Sequential([
                        iaa.Affine(
                            scale=(0.5, 1.2),
                            rotate=(-15, 15)
                        ),  # rotate the image
                        iaa.Flipud(0.5),
                        iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                        iaa.Sometimes(
                            0.1,
                            iaa.GaussianBlur((0.1, 1.5)),
                        ),
                        iaa.Sometimes(
                            0.1,
                            iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                        )
                    ])
        self.transforms = transforms.Compose([ToTensor()])

    def __getitem__(self, index):

        sample = {}
        if self.mode == 'train':
            
            mid_slice = self.image.shape[1] // 2

            image = self.image[index][mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
            label = self.label[index][mid_slice,...].copy() # H*W

            sample['origin_image'] = image
            sample['origin_label'] = label
            
            image = image.transpose(1, 2, 0)
            segmap = SegmentationMapsOnImage(np.uint8(label), shape=(256, 256))

            # data augmentation
            image, label = self.data_aug(image=image, segmentation_maps=segmap)

            image, label = image.copy(), label.copy()

            image = image.transpose(2, 0, 1)
            label = label.get_arr()
        
        elif self.mode == 'val':
            mid_slice = self.image.shape[1] // 2
            image = self.image[index][mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
            label = self.label[index][mid_slice, ...].copy()
            image, label = torch.from_numpy(image), torch.from_numpy(label)

        else:
            return self.load_test_data(index)

        sample['image'] = image
        sample['label'] = label
        
        if self.mode == 'train':
            sample = self.transforms(sample)
        
        return sample
        
    def __len__(self):
        if self.mode != 'test':
            return self.image.shape[0]
        else:
            return self.datasize
        
    def _get_image_list(self, filename_list):
        
        if self.mode == 'train':
            if os.path.exists(self.train_data_path):
                print("load train data from h5 file...")
                f = h5py.File(self.train_data_path, 'r')
                return f['image'][:], f['label'][:]
            else:
                print("train_data not existed...")
        else:
            if os.path.exists(self.val_data_path):
                print("load val data from h5 file...")
                f = h5py.File(self.val_data_path, 'r')
                return f['image'][:], f['label'][:]
            else:
                print("val_data not existed...")

        ct_list, label_list = [], []

        for dic in filename_list:

            data = np.load(dic['preprocess_npy'])
            ct, label = data[0], data[-1]
            for i in range(ct.shape[1]):
                ct[:,i,...] = normalize_minmax_data(ct[:,i,...]).copy()
            
            ct_list.extend(ct)
            label_list.extend(label)
            # if self.mode == 'train':
            #     ct_list.extend(data[0])
            #     label_list.extend(data[-1])
            # else:
            #     ct_list.append(data[0])
            #     label_list.append(data[-1])

        ct_arr = np.array(ct_list)
        label_arr = np.array(label_list)

        if self.mode == 'train':
            file = h5py.File(self.train_data_path, 'w')
        else:
            file = h5py.File(self.val_data_path, 'w')

        file['image'] = ct_arr
        file['label'] = label_arr
        file.close()

        print('save ', self.mode, ' data done...')
        
        return ct_arr, label_arr

    def _scan_test_data(self, filename_list):

        index2pid_dict = {}
        index2slice_dict = {}
        patient_id_list = []
        cur_ind = 0
        for dic in filename_list:
            pid = dic['preprocess_npy'].split('/')[-1].split('.')[0][-3:]
            patient_id_list.append(pid)
            data = np.load(dic['preprocess_npy'])
            num_slices = data.shape[1]
            for cnt in range(num_slices):
                index2pid_dict[cur_ind] = pid
                index2slice_dict[cur_ind] = cnt
                cur_ind += 1
        datasize = cur_ind
        return datasize, patient_id_list, index2pid_dict, index2slice_dict
    

    def load_test_data(self, index):

        patient_id, slice_id = self.find_pid_slice_id(index)
        data = np.load(self.filename_list[self.pid_map[patient_id]]['preprocess_npy'])
        image_3d, label_3d = data[0], data[-1]

        image, label = image_3d[slice_id], label_3d[slice_id]
        
        mid_slice = image.shape[0] // 2
        
        image = image[mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
        label = label[mid_slice, ...].copy()

        return {
            'image' : torch.from_numpy(image),
            'label' : torch.from_numpy(label).squeeze(),
            'pid' : patient_id
        }

    def find_pid_slice_id(self, index):
        '''
        given an index, find the patient id and slice id
        return the current id
        :return:
        '''
        self.p_id = self.index2pid_dict[index]
        self.slice_id = self.index2slice_dict[index]

        return self.p_id, self.slice_id


    def get_patient_data_for_testing(self, pid_index, crop_size=None, normalize_2D=False):

        '''
        prepare test volumetric data
        :param pad_size:[H',W']
        :param crop_size: [H',W']
        :return:
        data dict:
        {'image':torch tensor data N*3*H'*W'
        'label': torch tensor data: N*H'*W'
        }
        '''
        self.p_id = self.patient_id_list[pid_index]

        data = np.load(self.filename_list[self.pid_map[self.p_id]]['preprocess_npy'])
        image, label = data[0], data[-1]

        for i in range(image.shape[1]):
            image[:,i,...] = normalize_minmax_data(image[:,i,...]).copy()

        mid_slice = image.shape[1] // 2
        
        image = image[:, mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
        label = label[:, mid_slice, ...].copy()
        
        image_tensor = torch.from_numpy(image).float()
        label_tensor = torch.from_numpy(label).squeeze().long()
        
        dict = {
            'image': image_tensor,
            'label': label_tensor
        }
        return dict
    
    def get_id(self):
        return self.p_id
    
    def get_voxel_spacing(self):
        return [1., 1., 1.]


def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


class ToTensor(object):
    """Convert ndarrays in sample to Tensors.
    
        return:
        image : c * h * w
        label: h * w
    """

    def __call__(self, sample, maskresize=None, imageresize=None):
        
        image, label = sample['image'], sample['label']
        if len(label.shape) == 2:
            label = label.reshape((1,)+label.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,)+image.shape)
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).squeeze().float(),
                'origin_image': torch.from_numpy(sample['origin_image']).float(),
                'origin_label': torch.from_numpy(sample['origin_label']).squeeze().float()
        }


class RandomDataset(Dataset):
    
    def __init__(self):
        super(RandomDataset, self).__init__()

    def __getitem__(self, index):
        
        label = torch.rand([256, 256])
        label[label>0.5] = 1
        label[label<=0.6] = 0

        sample = {}
        sample['image'] = torch.rand([3, 256, 256])
        sample['label'] = label
        return sample

    def __len__(self):
        return 2000
    


