from locale import normalize
from multiprocessing import reduction
import pdb
from turtle import pd
import numpy as np
import torch.nn as nn
import torch
import random
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def context_mask_label_yuan(label, img):
    batch_size, img_x, img_y, img_z = label.shape[0], label.shape[1], label.shape[2], label.shape[3]

    # 存储所有值为 1 的像素坐标
    all_nonzero_indices = label.nonzero()

    if all_nonzero_indices.size(0) == 0:
        # 这里假设 context_mask 已经定义，否则需要提供它的实现
        return context_mask(img, 2 / 3)

    # 计算所有值为 1 的像素坐标的最小和最大坐标
    min_coord = all_nonzero_indices.min(dim=0)[0][1:]
    max_coord = all_nonzero_indices.max(dim=0)[0][1:]

    # 计算平均坐标的中心
    avg_center = (min_coord + max_coord) // 2

    # 计算能完全覆盖的圆形半径
    radius = max(max_coord - min_coord) // 2

    # 扩大 20% 半径
    new_radius = int(radius * 1.2)

    # 使用 torch.meshgrid 生成网格坐标
    x, y, z = torch.meshgrid(torch.arange(img_x), torch.arange(img_y), torch.arange(img_z), indexing='ij')
    x = x.to(label.device)
    y = y.to(label.device)
    z = z.to(label.device)

    # 计算每个坐标到中心的距离的平方
    dist_squared = (x - avg_center[0]) ** 2+(y - avg_center[1]) ** 2+(z - avg_center[2]) ** 2

    # 生成圆形的 mask
    mask = (dist_squared > new_radius ** 2).to(torch.long)
    loss_mask = torch.cat([mask.unsqueeze(0)])

    return mask, loss_mask

def context_mask_label(label, img):
    """
    从 sampled_batch['label'] 中计算每张图像中值为 1 的区域的最大坐标和最小坐标，
    然后分别计算所有最大坐标和最小坐标的平均值，并以此生成一个 mask。
    如果平均坐标生成的区域小于 64x64x64，则扩展到该大小。
    
    Args:
    - sampled_batch (dict): 包含 'image' 和 'label' 的字典
    - region_size (int): 最小区域大小，默认为 64
    
    Returns:
    - mask (torch.Tensor): 掩码，形状与 label_batch 相同
    """
    region_size=64
    batch_size, img_x, img_y, img_z = label.shape[0],label.shape[1],label.shape[2],label.shape[3]  # 获取标签的形状
    # print('1111111111',batch_size) # 8
    # 存储每张图像的最小和最大坐标
    all_min_coords = []
    all_max_coords = []
    
    for b in range(batch_size):
        current_label = label[b]  # 当前图像的标签
        
        # 获取所有值为 1 的像素的坐标
        nonzero_indices = current_label.nonzero()  # shape: (num_nonzero, 3)
        
        if nonzero_indices.size(0) > 0:
            # 计算当前图像的最小和最大坐标
            min_coord = nonzero_indices.min(dim=0)[0]  # 最小坐标
            max_coord = nonzero_indices.max(dim=0)[0]  # 最大坐标
        
        
            # 存储坐标
            all_min_coords.append(min_coord)
            all_max_coords.append(max_coord)
            # print('all_min_coords = ',all_min_coords)
            # print('all_max_coords = ',all_max_coords)
    if len(all_min_coords) == 0 or len(all_max_coords) == 0:
        return context_mask(img, 2/3)
            
    avg_min_coord = torch.mean(torch.stack(all_min_coords).float(), dim=0).long()
    avg_max_coord = torch.mean(torch.stack(all_max_coords).float(), dim=0).long()
    # print('avg_min_coord = ',avg_min_coord)
    # print('avg_max_coord = ',avg_max_coord)
    # 计算最小坐标和最大坐标的区域大小
    current_size = avg_max_coord - avg_min_coord
    
    # 计算平均坐标的中心
    avg_center = (avg_min_coord + avg_max_coord) // 2
    
    # 如果区域小于指定的 region_size，则扩展该区域
    if current_size[0] < region_size:
        expand_size = region_size - current_size[0]
        avg_min_coord[0] -= expand_size // 2  # 向前扩展
        avg_max_coord[0] += (expand_size + 1) // 2  # 向后扩展

    if current_size[1] < region_size:
        expand_size = region_size - current_size[1]
        avg_min_coord[1] -= expand_size // 2
        avg_max_coord[1] += (expand_size + 1) // 2

    if current_size[2] < region_size:
        expand_size = region_size - current_size[2]
        avg_min_coord[2] -= expand_size // 2
        avg_max_coord[2] += (expand_size + 1) // 2
    
    # 确保扩展后的坐标不超过图像的边界
    avg_min_coord = torch.clamp(avg_min_coord, min=torch.tensor([0, 0, 0], device=avg_min_coord.device))
    avg_max_coord = torch.clamp(avg_max_coord, max=torch.tensor([img_x, img_y, img_z], device=avg_max_coord.device))
    
    loss_mask = torch.ones(1, img_x, img_y, img_z).to(device)
    # print('2222222222222')
    # print(loss_mask.shape)
    mask = torch.ones(img_x, img_y, img_z).to(device)
    mask[avg_min_coord[0]:avg_max_coord[0], avg_min_coord[1]:avg_max_coord[1], avg_min_coord[2]:avg_max_coord[2]] = 0
    loss_mask[:, avg_min_coord[0]:avg_max_coord[0], avg_min_coord[1]:avg_max_coord[1], avg_min_coord[2]:avg_max_coord[2]] = 0
    # print(loss_mask.shape)
    return mask.long(), loss_mask.long()


def context_mask(img, mask_ratio):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    # print('--------------',batch_size)
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).to(device)
    # print('333333333333')
    # print(loss_mask.shape)
    mask = torch.ones(img_x, img_y, img_z).to(device)
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, 112 - patch_pixel_x)
    h = np.random.randint(0, 112 - patch_pixel_y)
    z = np.random.randint(0, 80 - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()

def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x/3)+1, int(patch_pixel_y/3)+1, int(patch_pixel_z/3)
    size_x, size_y, size_z = int(img_x/3), int(img_y/3), int(img_z/3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs*size_x, (xs+1)*size_x - mask_size_x - 1)
                h = np.random.randint(ys*size_y, (ys+1)*size_y - mask_size_y - 1)
                z = np.random.randint(zs*size_z, (zs+1)*size_z - mask_size_z - 1)
                mask[w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
                loss_mask[:, w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
    return mask.long(), loss_mask.long()

def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length -1)
    mask[:, :, z:z+z_length] = 0
    loss_mask[:, :, :, z:z+z_length] = 0
    return mask.long(), loss_mask.long()

def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    # print('111111111111111')
    # print(mask.shape) # [1, 96, 96, 96]
    # print(net3_output.shape) # [1, 2, 96, 96, 96]
    # print(img_l.shape) # [1, 96, 96, 96]
    dice_loss = DICE(net3_output, img_l, mask) * image_weight 
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def sup_loss(output, label):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label)
    loss_ce = torch.mean(CE(output, label))
    loss = (dice_loss + loss_ce) / 2
    return loss

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

@torch.no_grad()
def update_ema_students(model1, model2, ema_model, alpha):
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        ema_param.data.mul_(alpha).add_(((1 - alpha)/2) * param1.data).add_(((1 - alpha)/2) * param2.data)

@torch.no_grad()
def parameter_sharing(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data

class BBoxException(Exception):
    pass

def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 2)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 0)

    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))

def get_bbox_mask(mask):
    batch_szie, x_dim, y_dim, z_dim = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    mix_mask = torch.ones(batch_szie, 1, x_dim, y_dim, z_dim).cuda()
    for i in range(batch_szie):
        curr_mask = mask[i, ...].squeeze()
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bbox_3d(curr_mask)
        mix_mask[i, :, min_x:max_x, min_y:max_y, min_z:max_z] = 0
    return mix_mask.long()

