import os
import random
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.optim import optimizer
from torchvision.utils import save_image
import torch.nn.functional as F
from tqdm import tqdm
from datasets.mvtec import MVTecDataset, FSAD_Dataset_train, FSAD_Dataset_test
from utils.funcs import EarlyStop, denorm
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from utils.gen_mask import gen_mask
from models.fine_model import Encoder, Predictor
from models.coarse_model import ICA_net, fca_net
from losses.ssim_loss import SSIM_Loss
from losses.norm_loss import CosLoss, L2Loss
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from skimage import morphology, measure
from scipy.ndimage import gaussian_filter
# from scipy.spatial.distance import mahalanobis
from collections import OrderedDict
import warnings
import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology, measure
from skimage.segmentation import mark_boundaries
warnings.filterwarnings("ignore")

import pickle
from sklearn.utils.multiclass import type_of_target


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='Alignment anomaly detection')
    parser.add_argument('--obj', type=str, default='transistor')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path', type=str, default='/mnt/cfs/chaoqinhuang/MVTec/')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--lambda1', type=float, default=0, help='hyper-param for ICA in total loss')
    parser.add_argument('--lambda2', type=float, default=1, help='hyper-param for FAS in total loss')
    parser.add_argument('--ICA_lr', type=float, default=0.01, help='learning rate of ICA in SGD')
    parser.add_argument('--others_lr', type=float, default=0.0001, help='learning rate of others in SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=8, help='shot count')
    parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
    parser.add_argument('--stn_mode', type=str, default='rotation_scale',
                        help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')

    args = parser.parse_args()

    args.input_channel = 3

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str()
    args.save_dir = './vis_padim_mvtec/' + args.obj + '/'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    ICA = ICA_net().to(device)
    FCA = fca_net(args).to(device)
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)

    # freeze the resnet parameter
    # for name, param in FCA.named_parameters():
    #     if 'stn' not in name:
    #         param.requires_grad=False #固定参数
    # for name, param in FCA.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    models = [ICA, FCA, ENC, PRED]

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = FSAD_Dataset_train(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size, shot=args.shot, batch=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False, **kwargs)

    test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # start training
    fixed_fewshot_list = torch.load(f'./fixshot/{args.obj}/{args.shot}_{args.inferences}.pt')

    print_log(('---------Padim+Augmentation--------'), log)
    image_auc_list = []
    pixel_auc_list = []
    for inference_round in tqdm(range(args.inferences)):
        scores_list, test_imgs, gt_list, gt_mask_list = test(args, models, inference_round, fixed_fewshot_list, test_loader, **kwargs)
        # print("score list: ", scores_list)
        scores = np.asarray(scores_list)
        # Normalization
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

        # calculate image-level ROC AUC score
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)
        fpr, tpr, _ = roc_curve(gt_list, img_scores)
        img_roc_auc = roc_auc_score(gt_list, img_scores)
        # print('image ROCAUC: %.3f' % (img_roc_auc))
        image_auc_list.append(img_roc_auc)

        # calculate per-pixel level ROCAUC
        gt_mask = np.asarray(gt_mask_list)

        gt_mask = (gt_mask > 0.5).astype(np.int_)


        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]

        fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_auc_list.append(per_pixel_rocauc)

        plt.plot(fpr, tpr, label='%s pixel_ROCAUC: %.3f' % (args.obj, per_pixel_rocauc))
        plt.legend(loc="lower right")
        save_dir = args.save_dir + 'pictures_{:.4f}_{}'.format(threshold,inference_round)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, args.obj + '_roc_curve.png'), dpi=100)
        plt.close()

        plot_fig_new(args, test_imgs, scores, gt_mask_list, threshold, save_dir)
            

    image_auc_list = np.array(image_auc_list)
    pixel_auc_list = np.array(pixel_auc_list)

    mean_img_auc = np.mean(image_auc_list)
    mean_pixel_auc = np.mean(pixel_auc_list)

    print_log(('Test Result(img, pixel) padim: ({:.6f}, {:.6f})'.format(mean_img_auc, mean_pixel_auc)), log)
    log.close()

def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    # x = (x.transpose(1, 2, 0) * 255.).astype(np.uint8)
    return x

def plot_fig_new(args, test_img, scores, gts, threshold, save_dir):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    if not os.path.exists(os.path.join(save_dir, 'heat/')):
        os.makedirs(os.path.join(save_dir, 'heat/'))
    if not os.path.exists(os.path.join(save_dir, 'pre/')):
        os.makedirs(os.path.join(save_dir, 'pre/'))
    if not os.path.exists(os.path.join(save_dir, 'res/')):
        os.makedirs(os.path.join(save_dir, 'res/'))
    if not os.path.exists(os.path.join(save_dir, 'all/')):
        os.makedirs(os.path.join(save_dir, 'all/'))
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        #recon_img = recon_imgs[i]
        #recon_img = denormalization(recon_img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        #ax_img[1].imshow(recon_img)
        #ax_img[1].title.set_text('Reconst')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        fig_img.savefig(os.path.join(save_dir, 'all', args.obj + '_{}'.format(i)), dpi=100)
        plt.close()
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        plt.savefig(os.path.join(save_dir, 'heat', args.obj + '_{}'.format(i)))
        plt.close()
        plt.imshow(mask, cmap='gray')
        plt.savefig(os.path.join(save_dir, 'res', args.obj + '_{}'.format(i)))
        plt.close()
        plt.imshow(vis_img)
        plt.savefig(os.path.join(save_dir, 'pre', args.obj + '_{}'.format(i)))
        plt.close()

def test(args, models, cur_epoch, fixed_fewshot_list, test_loader, **kwargs):
    #ICA = models[0]
    FCA = models[1]
    ENC = models[2]
    PRED = models[3]

    #ICA.eval()
    FCA.eval()
    ENC.eval()
    PRED.eval()

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # support_dataset = MVTecDataset(args.data_path, class_name=args.obj, is_train=True, resize=args.img_size,
    #                              shot=args.shot)
    # support_data_loader = torch.utils.data.DataLoader(support_dataset, batch_size=args.shot, shuffle=True, **kwargs)
    # support_img = iter(support_data_loader).next()

    support_img = fixed_fewshot_list[cur_epoch]
    augment_support_img = support_img

    # rotate img with small angle
    for angle in [-np.pi/4, -3 * np.pi/16, -np.pi/8, -np.pi/16, np.pi/16, np.pi/8, 3 * np.pi/16, np.pi/4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)

    # translate img
    for a,b in [(0.2,0.2), (-0.2,0.2), (-0.2,-0.2), (0.2,-0.2), (0.1,0.1), (-0.1,0.1), (-0.1,-0.1), (0.1,-0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    
    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)

    # rgb to grey img
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)

    # rotate img in 90 degree
    for angle in [1,2,3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)

    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]


    # torch version
    with torch.no_grad():
        support_feat = FCA(augment_support_img.to(device))

    train_outputs['layer1'].append(FCA.stn1_output)
    train_outputs['layer2'].append(FCA.stn2_output)
    train_outputs['layer3'].append(FCA.stn3_output)

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = train_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name], True)

    # calculate multivariate Gaussian distribution
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    mean = torch.mean(embedding_vectors, dim=0)
    cov = torch.zeros(C, C, H * W).to(device)
    I = torch.eye(C).to(device)
    for i in range(H * W):
        cov[:, :, i] = torch.cov(embedding_vectors[:, :, i].T) + 0.01 * I
    # save learned distribution
    train_outputs = [mean, cov]

    # torch version
    query_imgs = []
    gt_list = []
    mask_list = []

    for (query_img, _, mask, y) in test_loader:
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())
        
        # model prediction
        query_feat = FCA(query_img.to(device))
        test_outputs['layer1'].append(FCA.stn1_output)
        test_outputs['layer2'].append(FCA.stn2_output)
        test_outputs['layer3'].append(FCA.stn3_output)

    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name], True)

    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    dist_list = []


    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = torch.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis_torch(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = torch.tensor(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    padim_score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(padim_score_map.shape[0]):
        padim_score_map[i] = gaussian_filter(padim_score_map[i], sigma=4)

    return padim_score_map, query_imgs, gt_list, mask_list

def save_snapshot(x, x2, model, save_dir, save_dir2, log):
    model.eval()
    with torch.no_grad():
        x_fake_list = x
        recon = model(x)
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(denorm(x_concat.data.cpu()), save_dir, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir)), log)

        x_fake_list = x2
        recon = model(x2)
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(denorm(x_concat.data.cpu()), save_dir2, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir2)), log)

if __name__ == '__main__':
    main()




