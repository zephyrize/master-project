import sys
sys.path.append('../../')

from networks.official_transUnet.vit_seg_modeling import VisionTransformer as TransUNet
from networks.official_transUnet.vit_seg_modeling import CONFIGS as TransUNet_CONFIGS
import numpy as np
import torch
from networks.utils_layers import measure_inference_speed


def get_transUnet(img_size=256, mode='train', model_name='R50-ViT-B_16'):

    transUnet_config = TransUNet_CONFIGS[model_name]

    transUnet_config.n_classes = 1
    
    if model_name.find('R50') != -1:
        transUnet_config.patches.grid = (int(256 // transUnet_config.patches.size[0]), 
                                             int(256 // transUnet_config.patches.size[1]))
    
    model = TransUNet(transUnet_config, img_size=img_size, num_classes=1)

    # if mode == 'train':
    #     if model_name == 'R50-ViT-B_16':
    #         model.load_from(weights=np.load('model/vit_checkpoint/R50+ViT-B_16.npz'))
    #     elif model_name == 'ViT-B_16':
    #         model.load_from(weights=np.load('model/vit_checkpoint/ViT-B_16.npz'))
    #     elif model_name == 'ViT-B_32':
    #         model.load_from(weights=np.load('model/vit_checkpoint/ViT-B_32.npz'))
    
    return model


from torchstat import stat

model = get_transUnet(img_size=256)
total_params = sum([param.nelement() for param in model.parameters()])


# print(model(torch.rand([1,3,256,256])).shape)
# stat(model, (3,256,256))
print("Number of parameter: %.2fM" % (total_params/1e6))


model = model.cuda()
model.eval()

print('fps: ', measure_inference_speed(model, (torch.randn(1, 3, 256, 256).cuda(),)))
    






