# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math

import torch.nn as nn
import torch.nn.functional as F

from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT
# from networks_other import init_weights
# from utils import UnetConv3, UnetUp3, UnetUp3_CT

class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)

class unet_3D_un(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_un, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # deep supervision
        self.dsv4 = UnetDsv3(
            in_size=filters[3], out_size=2, scale_factor=8)
        self.dsv3 = UnetDsv3(
            in_size=filters[2], out_size=2, scale_factor=4)
        self.dsv2 = UnetDsv3(
            in_size=filters[1], out_size=2, scale_factor=2)
        self.dsv1 = nn.Conv3d(
            in_channels=filters[0], out_channels=2, kernel_size=1)
        
        self.dropout1 = nn.Dropout3d(p=0.5)
        self.dropout2 = nn.Dropout3d(p=0.3)
        self.dropout3 = nn.Dropout3d(p=0.2)
        self.dropout4 = nn.Dropout3d(p=0.1)



    def forward(self, inputs):
        # print('inputs.shape = ',inputs.shape)

        conv1 = self.conv1(inputs)
        # print('conv1.shape = ',conv1.shape)
        maxpool1 = self.maxpool1(conv1)
        # print('maxpool1.shape = ',maxpool1.shape)

        conv2 = self.conv2(maxpool1)
        # print('conv2.shape = ',conv2.shape)
        maxpool2 = self.maxpool2(conv2)
        # print('maxpool2.shape = ',maxpool2.shape)

        conv3 = self.conv3(maxpool2)
        # print('conv3.shape = ',conv3.shape)
        maxpool3 = self.maxpool3(conv3)
        # print('maxpool3.shape = ',maxpool3.shape)

        conv4 = self.conv4(maxpool3)
        # print('conv4.shape = ',conv4.shape)
        maxpool4 = self.maxpool4(conv4)
        # print('maxpool4.shape = ',maxpool4.shape)

        center = self.center(maxpool4)
        # print('center.shape = ',center.shape)
        center = self.dropout1(center)

        
        up4 = self.up_concat4(conv4, center)
        up4 = self.dropout1(up4)
        # print('up4.shape = ',up4.shape)
        up3 = self.up_concat3(conv3, up4)
        up3 = self.dropout2(up3)
        # print('up3.shape = ',up3.shape)
        up2 = self.up_concat2(conv2, up3)
        up2 = self.dropout3(up2)
        # print('up2.shape = ',up2.shape)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout4(up1)
        # print('up1.shape = ',up1.shape)
        up1 = self.dropout2(up1)
        
        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        # final = self.final(up1)

        # return final
        return dsv1, dsv2, dsv3, dsv4

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
    
import h5py
import torch
# if __name__ == '__main__':
#     input = torch.rand(4, 1, 96, 96, 96)
#     net = unet_3D(n_classes=2, in_channels=1)
#     out = net(input)
#     # print(net)
#     print(out.shape)
#     n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     print("number of params: {:.2f}M".format(n_parameters/1024**2))
    
    
#     net.load_state_dict(torch.load('/home/zkx/SSL4MIS-master/model/BM_cps_120-unet_9/unet_3D/unet_3D_best_model1.pth'))
#     net.eval()
#     feature_maps = []  # 用于存储所有捕获的特征图
#     # 加载 h5 文件
#     h5_file_path = 'path/to/your_image.h5'
#     with h5py.File(h5_file_path, 'r') as h5_file:
#     # 假设图像存储在键名为 'image' 的数据集中
#         image_data = h5_file['image'][:]
    
# # 转换为 PyTorch Tensor
# # 假设图像是 3D 形式 [depth, height, width]
#     image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 增加 batch 维度和 channel 维度
#     print("Input tensor shape:", image_tensor.shape)  # 形状应为 [1, 1, depth, height, width]







    
    