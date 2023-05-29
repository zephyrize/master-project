import sys
sys.path.append('../')

from networks.SATNet import get_SATNet
import torch


model =  get_SATNet(img_size=256, dsv=False, sample_slices=7) # 在源码基础上改变后
pretrained_dict = torch.load("/data1/zfx/code/segBileDuct/model/BileDuct/SATNet_slices_7_bestModel.pt")  # 加载多卡训练后模型的字典
net_dict = model.state_dict()  # 当前模型的字典（单卡）
# 剔除多卡中的module关键字
for k, v in pretrained_dict.items():
    # 只删除一个
    name = k.replace('module.', '', 1)
    net_dict[name] = v
# 根据预训练字典更新当前模型
model.load_state_dict(net_dict)

new_path = "/data1/zfx/code/segBileDuct/model/BileDuct/SATNet_slices_7_withBestModel.pt"

torch.save(model.state_dict(), new_path)

print('transform done...')