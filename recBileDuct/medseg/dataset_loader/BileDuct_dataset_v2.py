import os
from os.path import join
import sys
sys.path.append('../../')
import json
from imgaug import augmenters as iaa
# from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import torch 
import h5py
import random
# from medseg.dataset_loader.base_segmentation_dataset import BaseSegDataset
from medseg.dataset_loader.dataset_utils import normalize_minmax_data
from medseg.dataset_loader.BileDuct_settings import get_Bile_split_policy
from medseg.dataset_loader.dataset_utils import normalize_minmax_data

DATASET_NAME = 'BileDuct'
IDX2CLASS_DICT = {
    0: 'BG',
    1: 'BILE',
}
DATA_FORMAT_NAME = 'BileDuct_{p_id}.npy'
IMAGE_SIZE = (256, 256, 3)
LABEL_SIZE = (256, 256)

class BileDuctDataset(Dataset):
    def __init__(self, 
                root_dir = '/data1/zfx/data/BileDuct/',
                split='train',
                transform=None,
                num_classes=2,
                image_size=IMAGE_SIZE,
                label_size=LABEL_SIZE,
                idx2cls_dict=IDX2CLASS_DICT,
                cval=0,
                keep_orig_image_label_pair=True,
                data_format_name=DATA_FORMAT_NAME,
                new_spacing=[1.0, 1.0, -1],
                normalize=True,
                sample_slices=3):
        

        self.split = split
        self.cval = cval
        self.frame = 'bileduct'
        self.transform = transform

        self.root_dir = join(root_dir, 'preprocessed_data')
        self.data_format_name = data_format_name
        self.normalize = normalize
        self.new_spacing = new_spacing

        self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict = self._scan_test_data(
            root_dir=self.root_dir,
            data_format_name=self.data_format_name,
            cval=self.cval,
            split=split
        )

        self.p_id = 0
        self.patient_number = len(self.patient_id_list)
        self.slice_id = 0
        self.index = 0
        self.keep_orig_image_label_pair = keep_orig_image_label_pair

        self.dataset_name = DATASET_NAME + '_{}'.format(split)
        if self.split == 'train':
            self.dataset_name += str(cval)
        print('load {},  containing {}, found {} slices'.format(
            self.dataset_name + self.frame, len(self.patient_id_list), self.datasize))

        self.sample_slices = sample_slices
        self.extend_slice = self.sample_slices // 2

        # self.voxelspacing = [1.36719, 1.36719, -1]

        # if self.new_spacing is not None:
        #     self.voxelspacing = self.new_spacing
        
    def __getitem__(self, index):

        data_dict = self._load_data(index)

        # print(data_dict['image'].shape, data_dict['label'].shape)
        data_dict = self._preprocess_data_to_tensors(
            data_dict['image'], data_dict['label']
        )
        return data_dict
        
    def __len__(self):
        return self.datasize
    
    def _load_data(self, index):
        
        '''
        give a index to fetch a data package for one patient
        :return:
        data from a patient.
        class dict: {
        'image': ndarray,H*W*CH, CH=1, for gray images
        'label': ndaray, H*W
        '''
        assert len(self.patient_id_list) > 0, "no data found in the disk at {}".format(
            self.root_dir)
        
        patient_id, slice_id = self.find_pid_slice_id(index)

        image_3d, label_3d = self.load_patientImage_from_nrrd(patient_id, normalize=self.normalize)

        max_id = image_3d.shape[0]
        id_list = list(np.arange(max_id))

        # remove slice w.o objects
        while True:
            # image = image_3d[slice_id]
            label = label_3d[slice_id]
            if abs(np.sum(label) - 0) > 1e-4:
                break
            else:
                id_list.remove(slice_id)
                random.shuffle(id_list)
                slice_id = id_list[0]

        # sample adjacent slices

        extend_slices = self.sample_slices // 2

        if slice_id >= extend_slices and slice_id < max_id - extend_slices:
            image = image_3d[slice_id-extend_slices:slice_id+extend_slices+1]
        elif slice_id < extend_slices:
            image = image_3d[0:slice_id+extend_slices+1]
            insert_slices_num = extend_slices - slice_id
            image = np.insert(image, [0]*insert_slices_num, image_3d[0], axis=0)
        elif slice_id+extend_slices >= max_id:
            image = image_3d[slice_id-extend_slices:max_id]
            insert_slices_num = extend_slices - (max_id-slice_id-1)
            image = np.insert(image, [image.shape[0]]*insert_slices_num, image_3d[max_id-1], axis=0)

        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
        
        cur_data_dict = {'image': image,
                         'label': label,
                         'pid': patient_id}
        
        return cur_data_dict

    def load_patientImage_from_nrrd(self, patient_id, new_spacing=None, normalize=False):
        '''
        get image_arr and label_arr from patient id\
        image: ndarray, N*H*W, N=slices number
        label: ndarray, N*H*W
        '''
        
        data_name = self.data_format_name.format(p_id=patient_id)
        data_path = join(self.root_dir, data_name)

        # load data
        img_arr, label_arr = self._load_img_label_from_path(data_path, new_spacing=new_spacing, normalize=normalize)

        return img_arr, label_arr

    def _load_img_label_from_path(self, data_path, new_spacing=None, normalize=False):

        '''
        load image and label by data_path
        :return:
        image: ndarray, N*H*W, N=slices number
        label: ndarray, N*H*W
        '''
        data = np.load(data_path)
        assert os.path.exists(data_path), print (f'{data_path} not found')
        
        mid_slice = data.shape[2]
        image, label = data[0][:,mid_slice//2,...].copy(), data[-1][:,mid_slice//2,...].copy()
        
        if self.normalize:
            image = normalize_minmax_data(image).copy()

        return image, label
    
    def _preprocess_data_to_tensors(self, image, label):

        '''
        data transform
        :return:
        dict {
        'image': torch tensor: C*H*W
        'label': torch tensor: H*W
        }
        '''

        assert len(image.shape) == 3 and len(
            label.shape) <= 3, 'input image and label dim should be 3 and 2 respectively, but got {} and {}'.format(
            len(image.shape),
            len(label.shape))

        # new_labels = self.formulate_labels(label)

        new_labels = np.uint8(label)
        orig_image = image
        orig_label = new_labels.copy()

        if_slice_data = True if len(label.shape) == 2 else False
        if if_slice_data:
            new_labels = new_labels[np.newaxis, :, :]
        new_labels = np.uint8(new_labels)

        if image.shape[0] > 1:
            new_labels = np.repeat(new_labels, axis=0, repeats=image.shape[0])
        
        transformed_image, transformed_label = self.transform(
            image, new_labels)
        if if_slice_data:
            transformed_label = transformed_label[0, :, :]
        
        assert transformed_image.shape[0] == self.sample_slices, "transformed image dim not equals to sample slices"
        
        result_dict = {
            'image': transformed_image,
            'label': transformed_label
        }

        if self.keep_orig_image_label_pair:
            orig_image_tensor = torch.from_numpy(orig_image).float()
            orig_label_tensor = torch.from_numpy(orig_label).long()
            result_dict['origin_image'] = orig_image_tensor
            result_dict['origin_label'] = orig_label_tensor

        return result_dict

    
    def _scan_test_data(self, root_dir, data_format_name, cval, split):
        
        # TODO: more operations
        patient_id_list = get_Bile_split_policy()[split]
        index2pid_dict = {}
        index2slice_dict = {}
        cur_ind = 0

        for pid in patient_id_list:
            
            data_path = join(root_dir, data_format_name.format(p_id=pid))

            if not os.path.exists(data_path):
                print(f'{data_path} not found')
                continue
            
            data = np.load(data_path)
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
                'label': torch.from_numpy(label).squeeze().float()}
    



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
    


if __name__ == '__main__':

    from medseg.dataset_loader.transform_bile import Transformations

    transform = Transformations(
        data_aug_policy_name='ACDC_affine_elastic_intensity'
    ).get_transformation()['train']

    train_set = BileDuctDataset(transform=transform)

    data = train_set.__getitem__(100)

    print(data['image'].shape, data['label'].shape)