#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random

import torch.nn as nn
from tqdm import tqdm
from commonDeA31 import get_pdn_medium, Student, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from torchvision import datasets
import cv2
import torchvision.models as models
from sklearn.metrics import roc_curve, roc_auc_score


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_loco',
                        choices=['mvtec_ad', 'mvtec_loco'])

    parser.add_argument('-o', '--output_dir', default=r'output/0707DeA31')
    parser.add_argument('-m', '--model_size', default='medium',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights',
                        default=r'/teacher_medium_final_state.pth')
    parser.add_argument('-s', '--sweights',
                        default=r'/mvtec_loco')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='/mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='/MVTec_LOCO',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=80000)
    return parser.parse_args()

device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# constants
seed = 3402
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])
def calculate_optimal_threshold(y_true, y_scores):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate Youden's J statistic
    J = tpr - fpr
    optimal_idx = np.argmax(J)  # Index of the optimal threshold
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold

def calculate_confusion_matrix(y_true, y_pred, threshold):
    # Apply threshold to obtain binary predictions

    y_pred_bin = [1 if x >= threshold else 0 for x in y_pred]

    # Initialize TP, TN, FP, FN
    TP, TN, FP, FN = 0, 0, 0, 0

    # Calculate TP, TN, FP, FN using the provided logic
    for i in range(len(y_true)):
        if y_true[i] == y_pred_bin[i] == 1:
            TP += 1
        elif y_true[i] == y_pred_bin[i] == 0:
            TN += 1
        elif y_true[i] == 1 and y_pred_bin[i] == 0:
            FN += 1
        elif y_true[i] == 0 and y_pred_bin[i] == 1:
            FP += 1

    return TP, TN, FP, FN

def calculate_metrics(TP, TN, FP, FN):
    # Sensitivity (SEN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Specificity (SPE)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    # Accuracy (ACC)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # F1-score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = sensitivity  # same as sensitivity
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'Sensitivity (SEN)': sensitivity,
        'Specificity (SPE)': specificity,
        'Accuracy (ACC)': accuracy,
        'F1-score': f1_score
    }
def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)
def cvt2heatmap(gray):

    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layer_ids=[0, 5, 10, 19, 28]):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        # 只保留到第30层的特征提取部分（可以根据需求调整）
        self.features = nn.Sequential(*list(vgg[:max(feature_layer_ids)+1]))
        # 关闭梯度计算
        for param in self.features.parameters():
            param.requires_grad = False
        self.feature_layer_ids = feature_layer_ids

        # 添加1x1卷积层，将通道数从384减少到3
        self.channel_reducer = nn.Conv2d(in_channels=384, out_channels=3, kernel_size=1)

    def forward(self, input, target):
        # 调整通道数到3
        input = self.channel_reducer(input)
        target = self.channel_reducer(target)

        input_features = []
        target_features = []
        x_in = input
        x_tg = target
        for i, layer in enumerate(self.features):
            x_in = layer(x_in)
            x_tg = layer(x_tg)
            if i in self.feature_layer_ids:
                input_features.append(x_in)
                target_features.append(x_tg)

        # 计算感知损失
        perceptual_loss = 0.0
        for f_in, f_tg in zip(input_features, target_features):
            perceptual_loss += torch.mean((f_in - f_tg) ** 2)

        return perceptual_loss
class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        # 调用父类的方法，获取图像和标签
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # 获取图像路径
        path = self.imgs[index][0]
        # 返回图像、标签和路径
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))



def main(adclass):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, adclass)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, adclass, 'test')
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, adclass, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, adclass, 'test'))
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                                  [train_size,
                                                                   validation_size],
                                                                  rng)
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, adclass, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.dataset')


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)

    # create models
    # if config.model_size == 'small':
    #     teacher = get_pdn_small(out_channels)
    #     student = get_pdn_small(2 * out_channels)
    # elif config.model_size == 'medium':
    teacher = get_pdn_medium(out_channels, padding=True)
    # else:
    #     raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    swgth=r"student_140000.pth"
    sweights = os.path.join(config.sweights, adclass)
    stweights = os.path.join(sweights, swgth)
    print(stweights)
    student = torch.load(stweights)


    if on_gpu:
        teacher.to(device)
        student.to(device)


    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)



    teacher.eval()
    student.eval()




    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))


def test(test_set, teacher, student, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir,
         desc='Running inference'):
    y_true = []
    y_score = []
    for image, target, path in test_set:
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.to(device)
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        anomaly_maptiffG = torch.nn.functional.interpolate(
            map_ae, (orig_height, orig_width), mode='bilinear')
        anomaly_maptiffG = anomaly_maptiffG[0, 0, :, :].to('cpu').detach().numpy()
        anomaly_maptiffG = gaussian_filter(anomaly_maptiffG, sigma=2)
        anomaly_maptiffL = torch.nn.functional.interpolate(
            map_st, (orig_height, orig_width), mode='bilinear')
        anomaly_maptiffL = anomaly_maptiffL[0, 0, :, :].to('cpu').detach().numpy()
        anomaly_maptiffL = gaussian_filter(anomaly_maptiffL, sigma=2)

        anomaly_maptiff = anomaly_maptiffG + anomaly_maptiffL

        ano_maptiffpng = min_max_norm(anomaly_maptiff) * 200
        ano_map = cvt2heatmap(ano_maptiffpng)

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            cv2.imwrite(
                file,
                anomaly_maptiff, ((int(cv2.IMWRITE_TIFF_RESUNIT), 2,
                                   int(cv2.IMWRITE_TIFF_COMPRESSION), 1,
                                   int(cv2.IMWRITE_TIFF_XDPI), 100,
                                   int(cv2.IMWRITE_TIFF_YDPI), 100)))
            filepng = file.replace("tiff", "png")
            filepng = filepng.replace("0707DeA31", "0707DeA31best")
            imgpng = cv2.imread(path)
            ano_map = show_cam_on_image(imgpng, ano_map)
            filepng = os.path.join("/data0/xmh/code/24code/EfficientAD/output", filepng)

            if not os.path.exists(os.path.dirname(filepng)):
                os.makedirs(os.path.dirname(filepng))
            cv2.imwrite(filepng, ano_map)
            map_combined = map_combined[0, 0].cpu().numpy()


        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)


    # Example usage:

    # Calculate optimal threshold based on AUC
    optimal_threshold = calculate_optimal_threshold(y_true, y_score)

    # Calculate TP, TN, FP, FN using the optimal threshold
    TP, TN, FP, FN = calculate_confusion_matrix(y_true, y_score, optimal_threshold)

    # Calculate metrics
    metrics = calculate_metrics(TP, TN, FP, FN)

    # Print metrics
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return auc * 100


@torch.no_grad()
def predict(image, teacher, student, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output_ae, autoencoder_output, student_output_st= student(image)
    # autoencoder_output = autoencoder(student_output_s)
    map_st = torch.mean((teacher_output - student_output_st) ** 2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output_ae) ** 2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(validation_loader, teacher, student,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in validation_loader:
        if on_gpu:
            image = image.to(device)
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end


@torch.no_grad()
def teacher_normalization(teacher, train_loader):
    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.to(device)
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    item_list = [
        #

        'pushpins',  # 1700*1000
        'juice_bottle',  # 800*1600
        'splicing_connectors',  # 1700*850
        'breakfast_box',  # 1600*1280
        'screw_bag',  # 1600*1100
    ]
    for i in item_list:
        print(i)
        main(i)