import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label

from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import BCP_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/zkx/SSL4MIS-master/data/Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='NRP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=2)

self_max_iterations = args.max_iterations

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    #target_softmax = F.softmax(target_logits, dim=1)
    mse_loss = (input_softmax - target_logits) ** 2
    return mse_loss

def get_XOR_region3(mixout, iter_num, max_iterations):
    outputs_soft = F.softmax(mixout, dim=1)
    log_probs = F.log_softmax(mixout, dim=1)
    uncertainty = -1.0 * torch.sum(outputs_soft * log_probs, dim=1, keepdim=True) 
    threshold1 = (0.5 + 0.5*ramps.sigmoid_rampup(iter_num, self_max_iterations))*np.log(2)
    diff_mask1 = (uncertainty > threshold1).float()
    outputs_soft = F.softmax(mixout, dim=1)
    confidences, _ = torch.max(outputs_soft, dim=1, keepdim=True)
    num_classes = 2
    percent = (iter_num + 1) / max_iterations
    cur_threshold = 1 / num_classes
    cur_threshold = percent * cur_threshold + (1 - percent) * confidences.mean()
    cur_threshold = torch.clip(cur_threshold, 0.5, 0.9)
    diff_mask2 = (confidences < cur_threshold).float() 
    M1 = (diff_mask1 * diff_mask2).float() + 0.6 * ((diff_mask1 + diff_mask2) == 1).float()
    return M1

def mix_mse_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask
    img_l_onehot = to_one_hot(img_l.unsqueeze(1), 2)
    patch_l_onehot = to_one_hot(patch_l.unsqueeze(1), 2)

    mse_loss = torch.mean(softmax_mse_loss(net3_output, img_l_onehot), dim=1) * mask * image_weight
    mse_loss += torch.mean(softmax_mse_loss(net3_output, patch_l_onehot), dim=1) * patch_mask * patch_weight


    loss = torch.sum(diff_mask * mse_loss) / (torch.sum(diff_mask) + 1e-16)
    return loss

voxel_kl_loss = nn.KLDivLoss(reduction="none")

def mix_max_kl_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False, diff_mask=None):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask
    with torch.no_grad():
        s1 = torch.softmax(net3_output, dim = 1)
        l1 = torch.argmax(s1, dim = 1)

        img_diff_mask = (l1 != img_l)
        patch_diff_mask = (l1 != patch_l)

        uniform_distri = torch.ones(net3_output.shape)
        uniform_distri = uniform_distri.to(device)
    
    kl_loss = torch.mean(voxel_kl_loss(F.log_softmax(net3_output, dim=1), uniform_distri), dim=1) * mask * img_diff_mask * image_weight
    kl_loss += torch.mean(voxel_kl_loss(F.log_softmax(net3_output, dim=1), uniform_distri), dim=1) * patch_mask * patch_diff_mask * patch_weight

    sum_diff = torch.sum(mask * img_diff_mask * diff_mask) + torch.sum(patch_mask * patch_diff_mask * diff_mask)

    loss = torch.sum(diff_mask * kl_loss) / (sum_diff + 1e-16)
    return loss

def load_net(net, path):
    # state = torch.load(str(path))
    state = torch.load(str(path), map_location='cpu')
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path), map_location='cuda:0')
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])
    

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).to(device)

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).to(device)
    

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    # return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    return  args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

import torch

def context_mask_label(label, img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    if torch.sum(label) == 0:
        return generate_mask(img)
    else:
        label_positions = (label > 0).nonzero(as_tuple=True)
        min_x = torch.min(label_positions[0]).item()
        max_x = torch.max(label_positions[0]).item()
        min_y = torch.min(label_positions[1]).item()
        max_y = torch.max(label_positions[1]).item()
        label_width = max_x - min_x
        label_height = max_y - min_y
        target_width = int(img_x * 2 / 3)
        target_height = int(img_y * 2 / 3)
        extra_width = max(0, target_width - label_width)
        extra_height = max(0, target_height - label_height)
        left_extra = extra_width // 2
        right_extra = extra_width - left_extra
        top_extra = extra_height // 2
        bottom_extra = extra_height - top_extra
        start_x = max(0, min_x - left_extra)
        end_x = min(img_x, max_x + right_extra)
        start_y = max(0, min_y - top_extra)
        end_y = min(img_y, max_y + bottom_extra)
        loss_mask = torch.zeros(batch_size, img_x, img_y).to(device)
        mask = torch.zeros(img_x, img_y).to(device)
        mask[start_x:end_x, start_y:end_y] = 1
        loss_mask[:, start_x:end_x, start_y:end_y] = 1

        return mask.long(), loss_mask.long()


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).to(device)
    mask = torch.ones(img_x, img_y).to(device)
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).to(device)
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).to(device)
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).to(device)
    mask = torch.ones(img_x, img_y).to(device)
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    # print('dataset = ',dataset)
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     

    model = BCP_net(in_chns=1, class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            # img_mask, loss_mask = generate_mask(img_a)
            img_mask, loss_mask = context_mask_label(label_batch, img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            #-- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_train(args ,pre_snapshot_path, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     
    model = BCP_net(in_chns=1, class_num=num_classes)
    ema_model = BCP_net(in_chns=1, class_num=num_classes, ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    load_net(ema_model, pre_trained_model)
    load_net_opt(model, optimizer, pre_trained_model)
    logging.info("Loaded from {}".format(pre_trained_model))

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()
    ema_model.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
                   
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(device), label_batch.to(device)
            lab_volume_batch = volume_batch[:args.labeled_bs]
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            lab_label_batch = label_batch[:args.labeled_bs]
            unlabeled_label_batch = label_batch[args.labeled_bs:]

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            with torch.no_grad():
                img_mask, loss_mask = context_mask_label(label_batch, lab_volume_batch) # [256, 256],[2, 256, 256]
            
            mix_img = lab_volume_batch * img_mask + unlabeled_volume_batch * (1 - img_mask)
            mix_out = model(mix_img)
            mix_out_soft = torch.softmax(mix_out, dim=1)
            ema_out = ema_model(unlabeled_volume_batch)
            ema_out_argmax = torch.argmax(ema_out, dim=1)
            loss_mix_dice, loss_mix_ce = mix_loss(mix_out, lab_label_batch, ema_out_argmax, loss_mask, u_weight=args.u_weight, unlab=True)
            loss_mix = 0.5 * (loss_mix_dice + loss_mix_ce)

            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_output, dim=1)

            loss_ce = ce_loss(outputs[:args.labeled_bs],label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            
            supervised_loss = 0.5 * (loss_dice + loss_ce)
            consistency_weight = get_current_consistency_weight(iter_num//150)
            # if iter_num < 1000:
            #     consistency_loss = 0.0
            # else:
            #     consistency_loss = torch.mean(
            #         (outputs_soft[args.labeled_bs:]-ema_output_soft)**2)
            # loss = supervised_loss + consistency_weight * consistency_loss
            
            
            # 基于Mbias的校正学习
            with torch.no_grad():
                diff_mask = get_XOR_region3(mix_out, iter_num, self_max_iterations)
            loss_mix_mse = mix_mse_loss(mix_out, lab_label_batch, ema_out_argmax.long(),  loss_mask, unlab=True, diff_mask=diff_mask)
            loss_mix_kl = mix_max_kl_loss(mix_out, lab_label_batch, ema_out_argmax.long(), loss_mask, diff_mask=diff_mask)
            loss = supervised_loss + consistency_weight * (loss_mix + loss_mix_mse + loss_mix_kl )
            
            
                

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            # update_model_ema(model, ema_model, 0.99)
            
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)
            # writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            # if iter_num % 20 == 0:
            #     image = net_input_u[1, 0:1, :, :]
            #     writer.add_image('train/Un_Image', image, iter_num)
            #     outputs = torch.argmax(torch.softmax(out_u, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/Un_Prediction', outputs[1, ...] * 50, iter_num)
            #     labs = unl_label[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/Un_GroundTruth', labs, iter_num)

            #     image_l = net_input_l[1, 0:1, :, :]
            #     writer.add_image('train/L_Image', image_l, iter_num)
            #     outputs_l = torch.argmax(torch.softmax(out_l, dim=1), dim=1, keepdim=True)
            #     writer.add_image('train/L_Prediction', outputs_l[1, ...] * 50, iter_num)
            #     labs_l = l_label[1, ...].unsqueeze(0) * 50
            #     writer.add_image('train/L_GroundTruth', labs_l, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "./model/NRP/Prostate_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/NRP/Prostate_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/NRP_train.py', self_snapshot_path)

    # Pre_train
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    # Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    


