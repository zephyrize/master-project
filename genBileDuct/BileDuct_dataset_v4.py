import sys
sys.path.append('../../')
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
from PIL import Image
from torchvision import transforms as TR


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
        return 100
    

def get_Bile_split_policy(identifier="standard", cval=0):
    
    assert cval < 5 and cval >= 0, 'only support five fold cross validation, but got {}'.format(cval)

    test_list = ["002", "010", "012"]

    if identifier == 'standard':
        # 18/3/3 for training and validation and test.
        training_list = ['001', '003', '004', '005', '006', '007', '008', '011', '014', '015', '017', '018', '019',
                         '020', '021', '022', '024', '025']
        validate_list = ['009', '016', '023']

        return {
            'name': str(identifier) + '_cv_' + str(cval),
            'train': training_list,
            'val': validate_list,
            'test': test_list,
            'unlabelled': [],
            'test+unlabelled': test_list
        }
    else:
        pass

'''
修改加载val data的模式
'''
DATASET_NAME = 'BileDuct'
DATA_FORMAT_NAME = 'BileDuct_{p_id}.npy'
IDX2CLASS_DICT = {
    0: 'BG',
    1: 'BILE',
}

class BileDuctDataset(Dataset):


    def __init__(self, 
                 root_dir = '/data1/zfx/data/BileDuct/',
                 split='train', 
                 sample_slices=3, 
                 mix_train_val=False,
                 cval=0,
                 keep_orig_image_label_pair=True,
                 normalize=False,
                 tv=False) :

        self.cval = cval
        self.split = split
        self.root_dir = root_dir
        self.mix_train_val = mix_train_val
        self.data_format_name = DATA_FORMAT_NAME
        self.normalize = normalize
        self.tv = tv

        self.train_data_path = join(self.root_dir, 'preprocessed_data', 'train_data.h5')
        self.val_data_path = join(self.root_dir, 'preprocessed_data', 'val_data.h5')
        self.test_data_path = join(self.root_dir, 'preprocessed_data', 'test_data.h5')
        self.filename_list = load_json(join(self.root_dir, 'preprocessed_data', "preprocess_dataset.json"))['test']
        # self.dataset_dir = join('/data1/zfx/data/', 'BileDuct') root_dir
        
        self.image, self.label, self.datasize, self.patient_id_list, self.index2pid_dict, self.index2slice_dict = self._scan_data(
            root_dir=join(self.root_dir, 'preprocessed_data'),
            data_format_name=self.data_format_name,
            cval=self.cval,
            split=self.split
        )

        if self.mix_train_val == True:
            assert split == 'train'
            self.image_val, self.label_val, self.datasize_val, self.patient_id_list_val, self.index2pid_dict_val, self.index2slice_dict_val = self._scan_data(
                root_dir=join(self.root_dir, 'preprocessed_data'),
                data_format_name=self.data_format_name,
                cval=self.cval,
                split='val'
            )
            self.datasize += self.datasize_val
        
        self.patient_number = len(self.patient_id_list)

        self.p_id = 0
        self.slice_id = 0
        self.index = 0
        self.keep_orig_image_label_pair = keep_orig_image_label_pair

        self.dataset_name = DATASET_NAME + '_{}'.format(split)

        self.sample_slices = sample_slices
        self.extend_slice = self.sample_slices // 2
        
        self.idx2cls_dict = {}

        for i in range(2):
            self.idx2cls_dict[i] = str(i)
        self.formalized_label_dict = self.idx2cls_dict
        # for test
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

        if self.split == 'test':
            return self.load_test_data(index)
        
        mid_slice = self.image.shape[1] // 2

        if self.mix_train_val == False:
            sample = self._get_train_or_val(index)
        else:
            train_data_size = self.datasize - self.datasize_val

            if index >= train_data_size:
                image = self.image_val[index%train_data_size][mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
                label = self.label_val[index%train_data_size][mid_slice,...].copy() # H*W
            else:
                image = self.image[index][mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
                label = self.label[index][mid_slice,...].copy() # H*W

            # sample['origin_image'], sample['origin_label'] = image, label

            # image, label = self._data_augmentation(image, label)

            sample['image'], sample['label'] = image, label
            
            # To Tensor
            sample = self.transforms(sample)

        # 加一个patient id
        p_id = self._get_pid(index)
        sample['pid'] = int(p_id)

        return sample

    def __len__(self):
        return self.datasize
    
    def _get_pid(self, index):
        if self.mix_train_val == False:
            p_id = self.index2pid_dict[index]
        else:
            if index + self.datasize_val < self.datasize:
                p_id = self.index2pid_dict[index]
            else:
                p_id = self.index2pid_dict_val[index%(self.datasize-self.datasize_val)]
        return p_id
    
    def get_id(self):
        return self.p_id
    
    def get_voxel_spacing(self):
        return [1., 1., 1.]

    def _get_train_or_val(self, index):

        sample = {}

        mid_slice = self.image.shape[1] // 2
        image = self.image[index][mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
        label = self.label[index][mid_slice,...].copy() # H*W

        # sample['origin_image'], sample['origin_label'] = image, label

        if self.split == 'train':
            pass
        else:
            image, label = torch.from_numpy(image), torch.from_numpy(label)

        sample['image'], sample['label'] = image, label

        if self.split == 'train':
            sample = self.transforms(sample)

        return sample
    
    def _data_augmentation(self, image, label):

        # data augmentation
        image = image.transpose(1, 2, 0)
        segmap = SegmentationMapsOnImage(np.uint8(label), shape=(256, 256))

        image, label = self.data_aug(image=image, segmentation_maps=segmap)
        image, label = image.copy(), label.copy()

        image = image.transpose(2, 0, 1)
        label = label.get_arr()
        
        return image, label

    
    def find_pid_slice_id(self, index):
        '''
        given an index, find the patient id and slice id
        return the current id
        :return:
        '''
        self.p_id = self.index2pid_dict[index]
        self.slice_id = self.index2slice_dict[index]

        return self.p_id, self.slice_id
    
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

        if self.normalize == True:
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
    
    
    def _scan_data(self, root_dir, data_format_name, cval, split):
        
        '''
        写的不是很优雅。。。
        '''
        # 先从h5文件里拿训练和验证数据
        if split == 'train':
            if os.path.exists(self.train_data_path):
                print("load train data from h5 file...")
                f = h5py.File(self.train_data_path, 'r')
                return f['image'][:], f['label'][:], f['datasize'].value, f.attrs['patient_id_list'], eval(f.attrs['index2pid_dict']), eval(f.attrs['index2slice_dict'])
            else:
                print("train_data not existed...")
        elif split == 'val':
            if self.tv == False:
                if os.path.exists(self.val_data_path):
                    print("load val data from h5 file...")
                    f = h5py.File(self.val_data_path, 'r')
                    return f['image'][:], f['label'][:], f['datasize'].value, f.attrs['patient_id_list'], eval(f.attrs['index2pid_dict']), eval(f.attrs['index2slice_dict'])
                else:
                    print("val_data not existed...")

            else:
                if os.path.exists(self.test_data_path):
                    print("load test data from h5 file...")
                    f = h5py.File(self.test_data_path, 'r')
                    return f['image'][:], f['label'][:], f['datasize'].value, f.attrs['patient_id_list'], eval(f.attrs['index2pid_dict']), eval(f.attrs['index2slice_dict'])
                else:
                    print("test_data not existed...")


        # print('begin process ', split, ' data')
        patient_id_list = get_Bile_split_policy()[split]
        if self.tv == True:
            patient_id_list = get_Bile_split_policy()['test']
        index2pid_dict = {}
        index2slice_dict = {}
        cur_ind = 0

        ct_list, label_list = [], []

        for pid in patient_id_list:
            
            data_path = join(root_dir, data_format_name.format(p_id=pid))

            if not os.path.exists(data_path):
                print(f'{data_path} not found')
                continue
            
            data = np.load(data_path)
            # num_slices = data.shape[1]
            num_slices = 0
            if self.split != 'test':
                ct, label = data[0], data[1]
                
                # TODO 在生成数据时，去除没有label的数据
                
                label_voxel_coords = np.where(label != 0)
                minzidx = int(np.min(label_voxel_coords[0]))
                maxzidx = int(np.max(label_voxel_coords[0])) + 1
                
                ct = ct[minzidx:maxzidx,...]
                label = label[minzidx:maxzidx,...]

                if self.normalize == True:
                    for i in range(ct.shape[1]):
                        ct[:,i,...] = normalize_minmax_data(ct[:,i,...]).copy()

                ct_list.extend(ct)
                label_list.extend(label)

                num_slices += ct.shape[0]

            for cnt in range(num_slices):
                index2pid_dict[cur_ind] = pid
                index2slice_dict[cur_ind] = cnt
                cur_ind += 1
            datasize = cur_ind


        if self.split == 'test':
            return None, None, datasize, patient_id_list, index2pid_dict, index2slice_dict

        ct_arr = np.array(ct_list)
        label_arr = np.array(label_list)

        if split == 'train':
            file = h5py.File(self.train_data_path, 'w')
        else:
            if self.tv == False:
                file = h5py.File(self.val_data_path, 'w')
            else:
                file = h5py.File(self.test_data_path, 'w')
        
        file['image'] = ct_arr
        file['label'] = label_arr
        file['datasize'] = datasize
        dtype = h5py.special_dtype(vlen=str)
        file.attrs.create('patient_id_list', patient_id_list, dtype=dtype)
        file.attrs.create('index2pid_dict', index2pid_dict, dtype=dtype) # 这里好像不需要用str，如果用的话，取值需要eval 
        file.attrs.create('index2slice_dict', index2slice_dict, dtype=dtype) # 同上

        file.close()

        print('save ', split if self.tv is False else 'test', ' data to h5 file done...')

        return ct_arr, label_arr, datasize, patient_id_list, index2pid_dict, index2slice_dict
        

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
        
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).squeeze().float()

        if image.shape[0] == 1:
            image = TR.functional.normalize(image, (0.5,), (0.5,))
        elif image.shape[0] == 3:
            image = TR.functional.normalize(image, (0.5,0.5,0.5), (0.5,0.5,0.5))
        return {'image': image,
                'label': label,
                # 'origin_image': torch.from_numpy(sample['origin_image']).float(),
                # 'origin_label': torch.from_numpy(sample['origin_label']).squeeze().float()
        }
    

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a


def normalize_minmax_data(image_data):
    """
    # 3D MRI scan is normalized to range between 0 and 1 using min-max normalization.
    Here, the minimum and maximum values are used as 2nd and 98th percentiles respectively from the 3D MRI scan.
    We expect the outliers to be away from the range of [0,1].
    input params :
        image_data : 3D MRI scan to be normalized using min-max normalization
    returns:
        final_image_data : Normalized 3D MRI scan obtained via min-max normalization.
    """
    min_val_2p = np.percentile(image_data, 2)
    max_val_98p = np.percentile(image_data, 98)
    final_image_data = np.zeros(
        (image_data.shape[0], image_data.shape[1], image_data.shape[2]), dtype=np.float32)
    # min-max norm on total 3D volume
    image_data[image_data < min_val_2p] = min_val_2p
    image_data[image_data > max_val_98p] = max_val_98p

    final_image_data = (image_data - min_val_2p) / \
        (1e-10 + max_val_98p - min_val_2p)

    return final_image_data

if __name__ == '__main__':

    dataset = BileDuctDataset(split='train', normalize=False)
    dataset = BileDuctDataset(split='val', normalize=False)

    print(len(dataset))




'''
train and val: 3438
train: 2794
val: 644
'''