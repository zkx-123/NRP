from networks.unet import UNet, UNet_2d, UNet_URPC
from networks.VNet import VNet
import torch.nn as nn
import torch
from networks.unet_3d import unet_3D
from networks.unet_3d_un import unet_3D_un
from networks.segmamba import SegMamba

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
    if net_type == "unet" and mode == "train":
        net = UNet(in_chns=in_chns, class_num=class_num).to(device)
    if net_type == "VNet" and mode == "train" and tsne==0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).to(device)
    if net_type == "unet_3D_un" and mode == "train" and tsne==0:
        net = unet_3D_un(in_channels=1, n_classes=2).to(device)
    if net_type == "unet_3D":
        net = unet_3D(in_channels=1, n_classes=2).to(device)
    if net_type == "segmamba":
        net = SegMamba(in_chans=1, out_chans=2, depths=[2,2,2,2], feat_size=[24, 48, 96, 192]).to(device)
    if net_type == "VNet" and mode == "test" and tsne==0:
        net = VNet(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    elif net_type == "unet_urpc":
        net = UNet_URPC(in_chns=in_chns, class_num=class_num).to(device)
    return net

def BCP_net(in_chns=1, class_num=2, ema=False):
    device1 = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    net = UNet_2d(in_chns=in_chns, class_num=class_num).to(device1)
    if ema:
        for param in net.parameters():
            param.detach_()
    return net

