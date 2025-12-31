import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm   
import pandas as pd
from networks.net_factory import net_factory
import matplotlib.pyplot as plt
from networks.unet1 import MCNet2d_v1
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/zkx/SSL4MIS-master/data/Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='NRP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self or pre')

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, net, case_dir, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (256 / x, 256 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().to(device)
        net.eval()
        with torch.no_grad():
            out_main = net(input)
            if len(out_main)>1:
                out_main=out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 256, y / 256), order=0)
            prediction[ind] = pred
    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    return first_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    snapshot_path = "./model/NRP/Prostate_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = "./model/NRP/Prostate_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path))

    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = np.zeros(4)
    
    individual_results = []
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        
        individual_results.append((case, first_metric))
    avg_metric = first_total / len(image_list)
    
    # txt
    with open(test_save_path+'../performance-xin-10-0.1.txt', 'w') as f:
        f.write("Individual results for each test image:\n")
        for case, metrics in individual_results:
            f.write(f"Case: {case}, Dice: {metrics[0]}, JC: {metrics[1]}, HD95: {metrics[2]}, ASD: {metrics[3]}\n")
        f.write("\nOverall average metrics:\n")
        f.write(f"Average Dice: {avg_metric[0]:.4f}, Average JC: {avg_metric[1]:.4f}, Average HD95: {avg_metric[2]:.4f}, Average ASD: {avg_metric[3]:.4f}\n")
    
    
    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
