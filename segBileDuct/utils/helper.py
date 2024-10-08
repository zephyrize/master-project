import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import nibabel as nib
import cv2

def get_pixels_hu(scans):
    #type(scans[0].pixel_array)
    #Out[15]: numpy.ndarray
    #scans[0].pixel_array.shape
    #Out[16]: (512, 512)
    # image.shape: (129,512,512)
    image = scans.pixel_array
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans.RescaleIntercept
    slope = scans.RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def transform_ctdata(image, windowWidth, windowCenter, normal=False):
        """
        注意，这个函数的self.image一定得是float类型的，否则就无效！
        return: trucated image according to window center and window width
        """
        minWindow = float(windowCenter) - 0.5*float(windowWidth)
        newimg = (image - minWindow) / float(windowWidth)
        newimg[newimg < 0] = 0
        newimg[newimg > 1] = 1
        if not normal:
            newimg = (newimg * 255).astype('uint8')
        return newimg


def show_img(res):
    plt.imshow(res, cmap ='gray')
    plt.show()


def nii2csv(args):


    FOLDER_IMAGE_ZDYY = args.data_root_dir + args.dataset + "/ZDYYLiverVesselNII/"
    FOLDER_MASK_ZDYY = args.data_root_dir + args.dataset + "/ZDYYLiverPredictNnunet20210401/"
    FOLDER_LABEL_ZDYY = args.data_root_dir + args.dataset + "/ZDYYLiverVesselNIIlabel20210826/"
    DATA_CSV_PATH = "data_path_info.csv"

    count = 0 # 计数
    list_name_case = []
    list_name_dataset = []
    list_path_image = []
    list_path_mask = []
    list_path_label = []
    list_train_or_test = []


    # 读取ZDYY数据集文件夹
    for root, dirs, files in os.walk(FOLDER_LABEL_ZDYY):
        for file in files: 
            file_path = os.path.join(root, file)
            idx_file = get_filename_number(file, 0)
            name_case = 'ZDYY_%(idx_file)03d'%{"idx_file": idx_file} # 给这个case起个名字
            path_image = FOLDER_IMAGE_ZDYY + 'ZDYY_%(idx_file)03d_0000.nii.gz'%{"idx_file": idx_file}
            path_mask = FOLDER_MASK_ZDYY + 'ZDYY_%(idx_file)03d.nii.gz'%{"idx_file": idx_file}

            list_name_case.append(name_case)
            list_name_dataset.append('ZDYY')
            list_path_image.append(path_image)
            list_path_mask.append(path_mask)
            list_path_label.append(file_path)
            count = count + 1
            list_train_or_test.append(int(count % 4 != 0))

    dataframe = pd.DataFrame({ \
        'name_case': list_name_case, \
        'name_dataset': list_name_dataset, \
        'raw_image_path':list_path_image, \
        'raw_mask_path':list_path_mask, \
        'raw_label_path':list_path_label, \
        'train_or_test': list_train_or_test})

    dataframe.to_csv(DATA_CSV_PATH, index=False, sep=',')




# torchvision helper 

def matplotlib_imshow(pre, label):
    pre[pre>=0.5] = 1
    pre[pre<0.5] = 0

    batch = pre.shape[0]

    plt.figure(figsize=(128, 128))
    
    for i in range(batch * 2):
        plt.subplot(batch//2, 4, i+1)

        if (i+1) %2 == 1:
            plt.imshow(label[i//2].permute(1,2,0).cpu().numpy(), cmap='gray')
            plt.title('label', fontsize=36)
        else:
            plt.imshow(pre[i//2].permute(1,2,0).cpu().numpy(), cmap='gray')
            plt.title('predict', fontsize=36)
        plt.axis('off')
        plt.subplots_adjust(left=0.15,bottom=0.1,top=0.5,right=0.25,hspace=0.2,wspace=0.25)    
    
    fig = plt.gcf()
    
    return fig


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector

    """
    _mask = [mask == i for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)

def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask


def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list


def get_nonzero_box(mask):

    mask_voxel_coords = np.where(mask != 0)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1])) 
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]



def test_single_volume(volume, model, batch_size, device, dsv=False, use_cpu=False, multi_loss=False):

    '''
    param: use_cpu : whether to use gpu for testing.
    '''


    '''
    2023.05.03 备注：
    玛德，为了适配CAT_Net的训练。这里做一些batch size的改动。
    草，真的无语，怎么会有人把batch size这个变量写到模型里面啊，纯脑瘫、傻逼。
    '''

    model.eval()
    '''
    return: PR: (z, c, x, y)
    '''
    PR = torch.Tensor([])
    
    if type(volume) is np.ndarray:
        volume = torch.from_numpy(volume)
        
    length = volume.shape[0]
    
    for i in range(0, length, batch_size):

        image = None
        flag = 0

        if i + batch_size <= length:
            image = volume[i:i+batch_size]
        else:
            flag = 1
            upper_slices_num = i + batch_size - length
            begin_slice = i
            image = volume[i-upper_slices_num:]
            assert image.shape[0] == batch_size

        if use_cpu is True:
            image = image.float()
        else:
            image = image.float().to(device)

        with torch.no_grad():
            output = model(image)
            
            infer = output[0] if isinstance(output, list) else output

            if use_cpu is True:
                PR = torch.cat([PR, infer], dim=0)  # .cpu().detach()
            else:
                PR = torch.cat([PR, infer.cpu().detach()], dim=0) 

            if flag == 1:
                for i in range(upper_slices_num):
                    PR = PR[torch.arange(PR.size(0))!=begin_slice]

    return PR


def get_device():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device




def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.5 * sigmoid_rampup(epoch, 50)

