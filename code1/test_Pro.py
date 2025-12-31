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
parser.add_argument('--exp', type=str, default='BCP', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
# mc-net
# parser.add_argument('--model', type=str, default='mcnet2d_v1', help='model_name')
# mc-net
# parser.add_argument('--model', type=str, default='unet_urpc', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--labelnum', type=int, default=7, help='labeled data')
# parser.add_argument('--stage_name', type=str, default='self_train-lab-M-xin-10-0.1-2', help='self or pre')
parser.add_argument('--stage_name', type=str, default='self_train-xin-10-0.1', help='self or pre')

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
    # 创建三个文件夹用于保存原图像、label 和预测结果
    image_dir = os.path.join(case_dir, 'images')
    label_dir = os.path.join(case_dir, 'labels')
    prediction_dir = os.path.join(case_dir, 'predictions')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
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

    # if np.sum(prediction == 2)==0:
    #     second_metric = 0,0,0,0
    # else:
    #     second_metric = calculate_metric_percase(prediction == 2, label == 2)

    # if np.sum(prediction == 3)==0:
    #     third_metric = 0,0,0,0
    # else:
    #     third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # 保存3D图像
    # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    # img_itk.SetSpacing((1, 1, 10))
    # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    # prd_itk.SetSpacing((1, 1, 10))
    # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    # lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    
    # # 可视化并保存
    # for ind in range(image.shape[0]):
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #     axes[0].imshow(image[ind], cmap='gray')
    #     axes[0].set_title('Image')
    #     axes[1].imshow(label[ind], cmap='gray')
    #     axes[1].set_title('Label')
    #     axes[2].imshow(prediction[ind], cmap='gray')
    #     axes[2].set_title('Prediction')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(test_save_path, f'{case}_slice_{ind}.png'))
    #     plt.close()
    # for ind in range(image.shape[0]):
    #     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #     axes[0].imshow(image[ind], cmap='gray')
    #     # axes[0].imshow(equalize_adapthist(image[ind]), cmap='gray')
    #     axes[0].set_title('Image')
    #     axes[1].imshow(label[ind], cmap='gray')
    #     axes[1].set_title('Label')
    #     axes[2].imshow(prediction[ind], cmap='gray')
    #     axes[2].set_title('Prediction')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(case_dir, f'{case}_slice_{ind}.png'))
    #     plt.close()
    # for ind in range(image.shape[0]):
    #     # 保存原图像
    #     plt.imsave(os.path.join(image_dir, f'{case}_slice_{ind}.png'), image[ind], cmap='gray')
    #     # 增强后的原图像
    #     # plt.imsave(os.path.join(image_dir, f'{case}_slice_{ind}.png'), equalize_adapthist(image[ind]), cmap='gray')
    #     # 保存 label
    #     plt.imsave(os.path.join(label_dir, f'{case}_slice_{ind}.png'), label[ind], cmap='gray')
    #     # 保存预测结果
    #     plt.imsave(os.path.join(prediction_dir, f'{case}_slice_{ind}.png'), prediction[ind], cmap='gray')
    
    
    # return first_metric, second_metric, third_metric
    return first_metric


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])
    # # semi
    snapshot_path = "./model-P-xin/BCP/Prostate_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
    test_save_path = "./model-P-xin/BCP/Prostate_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
    # fs
    # snapshot_path = "/home/zkx/SSL4MIS-master/model-P/Prostate/Fully_Supervised_{}_labeled-1/{}".format(FLAGS.labelnum, FLAGS.model)
    # test_save_path = "/home/zkx/SSL4MIS-master/model-P/Prostate/Fully_Supervised_{}_labeled-1/{}_predictions/".format(FLAGS.labelnum, FLAGS.model)
    
    # baseline
    # snapshot_path = "/home/zkx/SSL4MIS-master/model-P/Prostate/Fully_Supervised_35_labeled/{}".format(FLAGS.model)
    # test_save_path = "/home/zkx/SSL4MIS-master/model-P/Prostate/Fully_Supervised_35_labeled/{}_predictions/".format(FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)
    # net = MCNet2d_v1(in_chns=1, class_num=2).to(device)
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    # save_model_path = os.path.join(snapshot_path, 'iter_4400_dice_0.7815.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path))

    print("init weight from {}".format(save_model_path))
    net.eval()

    # first_total = 0.0
    first_total = np.zeros(4)
    # second_total = 0.0
    # third_total = 0.0
    
    individual_results = []
    for case in tqdm(image_list):
        # first_metric, second_metric, third_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        
        individual_results.append((case, first_metric))
        # second_total += np.asarray(second_metric)
        # third_total += np.asarray(third_metric)
    # avg_metric = [first_total / len(image_list), second_total / len(image_list), third_total / len(image_list)]
    avg_metric = first_total / len(image_list)
    
    # txt
    with open(test_save_path+'../performance-xin-10-0.1.txt', 'w') as f:
        f.write("Individual results for each test image:\n")
        for case, metrics in individual_results:
            f.write(f"Case: {case}, Dice: {metrics[0]}, JC: {metrics[1]}, HD95: {metrics[2]}, ASD: {metrics[3]}\n")
        f.write("\nOverall average metrics:\n")
        # f.write(f"Average Dice: {avg_metric[0][0]}, Average JC: {avg_metric[0][1]}, Average HD95: {avg_metric[0][2]}, Average ASD: {avg_metric[0][3]}\n")
        f.write(f"Average Dice: {avg_metric[0]:.4f}, Average JC: {avg_metric[1]:.4f}, Average HD95: {avg_metric[2]:.4f}, Average ASD: {avg_metric[3]:.4f}\n")
    
    # # execl
    # # 创建包含每个测试图像结果的数据框
    # columns = ['Case', 'Dice', 'JC', 'HD95', 'ASD']
    # individual_df = pd.DataFrame(individual_results, columns=columns)

    # # 创建包含总平均结果的数据框
    # overall_df = pd.DataFrame([['Overall Average', *avg_metric]], columns=columns)

    # # 将两个数据框合并
    # combined_df = pd.concat([individual_df, overall_df], ignore_index=True)

    # # 保存为 Excel 文件
    # excel_path = test_save_path + 'performance-0.5-0.6-xin2.xlsx'
    # combined_df.to_excel(excel_path, index=False)
    
    
    return avg_metric, test_save_path


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    # # print((metric[0]+metric[1]+metric[2])/3)
    # with open(test_save_path+'../performance-0.6-0.6-0.05-0.1.txt', 'w') as f:
    #     f.writelines('metric is {} \n'.format(metric))
    #     # f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
