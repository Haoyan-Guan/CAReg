import os
import random
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test, FSAD_all_Dataset_train
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from models.siamese import Encoder, Predictor
from models.stn import stn_net, FeatureExtractor
from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings

from utils.utils import KCenterGreedy, AnomalyMapGenerator, kCenterGreedy2
from typing import Dict
from torch import Tensor

from sklearn.random_projection import SparseRandomProjection
import faiss
import cv2

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def embedding_concat2(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)
    return z

def reshape_embedding2(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list

def generate_embedding(features: Dict[str, Tensor], layers) -> torch.Tensor:
        """Generate embedding from hierarchical feature map.

        Args:
            features: Hierarchical feature map from a CNN (ResNet18 or WideResnet)
            features: Dict[str:Tensor]:

        Returns:
            Embedding vector
        """

        embeddings = features[layers[0]]
        for layer in layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings

def reshape_embedding(embedding: Tensor) -> Tensor:
    """Reshape Embedding.

    Reshapes Embedding to the following format:
    [Batch, Embedding, Patch, Patch] to [Batch*Patch*Patch, Embedding]

    Args:
        embedding (Tensor): Embedding tensor extracted from CNN features.

    Returns:
        Tensor: Reshaped embedding tensor.
    """
    embedding_size = embedding.size(1)
    embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
    return embedding

def main():
    parser = argparse.ArgumentParser(description='Registration based Few-Shot Anomaly Detection')
    parser.add_argument('--obj', type=str, default='connector')
    parser.add_argument('--data_type', type=str, default='mvtec')
    parser.add_argument('--data_path_train', type=str, default='../Dataset/MVTec')
    parser.add_argument('--data_path_test', type=str, default='../Dataset/MPDD')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    #32, 4 for 8 shot.
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of others in SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
    # test in mvtec
    # parser.add_argument('--stn_mode', type=str, default='rotation_scale',
    #                     help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
    # test in MPDD
    parser.add_argument('--stn_mode', type=str, default='affine',
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
    if 'MVTec' in args.data_path_train:
        args.save_dir = './logs_mvtectrain/'
    elif 'MPDD' in args.data_path_train:
        args.save_dir = './logs_mpddtrain/'
    else:
        print('error fold!!!!!')
    #args.save_dir = './tmp/'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #args.save_model_dir = './logs_mvtec/' + args.stn_mode + '/' + str(args.shot) + '/' + args.obj + '/'
    args.save_model_dir = './logs_mpdd2/' + args.stn_mode + '/' + str(args.shot) + '/' + args.obj + '/'
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    log = open(os.path.join(args.save_dir, 'log_{}_{}.txt'.format(str(args.shot),args.obj)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    #STN = stn_net(args).to(device)
    '''
    #patchcore1
    backbone = 'wide_resnet50_2'
    pre_trained = True
    layers = ['layer2','layer3']
    STN = FeatureExtractor(backbone=backbone, pre_trained=pre_trained, layers=layers).to(device)
    '''
    #patchcore2
    STN = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True).to(device)
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)

    print(STN)

    STN_optimizer = optim.SGD(STN.parameters(), lr=args.lr, momentum=args.momentum)
    ENC_optimizer = optim.SGD(ENC.parameters(), lr=args.lr, momentum=args.momentum)
    PRED_optimizer = optim.SGD(PRED.parameters(), lr=args.lr, momentum=args.momentum)
    models = [STN, ENC, PRED]
    optimizers = [STN_optimizer, ENC_optimizer, PRED_optimizer]
    init_lrs = [args.lr, args.lr, args.lr]

    print('Loading Datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset = FSAD_all_Dataset_train(args.data_path_train, class_name=None, is_train=True, resize=args.img_size, shot=args.shot, batch=args.batch_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    test_dataset = FSAD_Dataset_test(args.data_path_test, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    # start training
    save_name = os.path.join(args.save_model_dir, '{}_{}_{}_model.pt'.format(args.obj, args.shot, args.stn_mode))
    start_time = time.time()
    epoch_time = AverageMeter()
    img_roc_auc_old = 0.0
    per_pixel_rocauc_old = 0.0
    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'./support_set/{args.obj}/{args.shot}_{args.inferences}.pt')
    print_log((f'---------{args.stn_mode}--------'), log)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizers, init_lrs, epoch, args)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        #train_patchcore2(models, epoch, train_loader, optimizers, log)
        if epoch <= args.epochs:
            image_auc_list = []
            pixel_auc_list = []
            for inference_round in tqdm(range(args.inferences)):
                gt_list_px_lvl, gt_list_img_lvl, gt_list, gt_mask_list = test_patchcore2(models, inference_round, fixed_fewshot_list,
                                                                     test_loader, **kwargs)
                '''
                scores = np.asarray(scores_list)
                # Normalization
                max_anomaly_score = scores.max()
                min_anomaly_score = scores.min()
                scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)

                # calculate image-level ROC AUC score
                img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
                gt_list = np.asarray(gt_list)

                img_roc_auc = roc_auc_score(gt_list, img_scores)
                image_auc_list.append(img_roc_auc)

                # calculate per-pixel level ROCAUC
                gt_mask = np.asarray(gt_mask_list)
                gt_mask = (gt_mask > 0.5).astype(np.int_)
                per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
                pixel_auc_list.append(per_pixel_rocauc)
                '''
                #print('gt_list', np.any(np.isnan(np.asarray(gt_list))), np.all(np.isfinite(np.asarray(gt_list))))
                #print('gt_list_img_lvl', np.any(np.isnan(np.asarray(gt_list_img_lvl))), np.all(np.isfinite(np.asarray(gt_list_img_lvl))))
                if np.any(np.isnan(np.asarray(gt_list_img_lvl))):
                    print('img_lvl nan')
                    continue

                img_roc_auc = roc_auc_score(gt_list, gt_list_img_lvl)
                image_auc_list.append(img_roc_auc)

                scores = np.asarray(gt_list_px_lvl)
                # Normalization
                max_anomaly_score = scores.max()
                min_anomaly_score = scores.min()
                scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
                gt_mask = np.asarray(gt_mask_list)
                gt_mask = (gt_mask > 0.5).astype(np.int_)
                per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
                pixel_auc_list.append(per_pixel_rocauc)

            image_auc_list = np.array(image_auc_list)
            pixel_auc_list = np.array(pixel_auc_list)
            mean_img_auc = np.mean(image_auc_list, axis = 0)
            mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)

            if mean_img_auc + mean_pixel_auc > per_pixel_rocauc_old + img_roc_auc_old:
                state = {'STN': STN.state_dict(), 'ENC': ENC.state_dict(), 'PRED':PRED.state_dict()}
                torch.save(state, save_name)
                per_pixel_rocauc_old = mean_pixel_auc
                img_roc_auc_old = mean_img_auc
            print('Img-level AUC:',img_roc_auc_old)
            print('Pixel-level AUC:', per_pixel_rocauc_old)

            print_log(('Test Epoch(img, pixel): {} ({:.6f}, {:.6f}) best: ({:.3f}, {:.3f})'
            .format(epoch-1, mean_img_auc, mean_pixel_auc, img_roc_auc_old, per_pixel_rocauc_old)), log)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        train_patchcore2(models, epoch, train_loader, optimizers, log)
        train_dataset.shuffle_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
        
    log.close()

def train(models, epoch, train_loader, optimizers, log):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN_optimizer = optimizers[0]
    ENC_optimizer = optimizers[1]
    PRED_optimizer = optimizers[2]

    STN.train()
    ENC.train()
    PRED.train()

    total_losses = AverageMeter()

    for (query_img, support_img_list, _) in tqdm(train_loader):
        STN_optimizer.zero_grad()
        ENC_optimizer.zero_grad()
        PRED_optimizer.zero_grad()

        query_img = query_img.squeeze(0).to(device)
        query_feat = STN(query_img)
        support_img = support_img_list.squeeze(0).to(device)
        B,K,C,H,W = support_img.shape

        support_img = support_img.view(B * K, C, H, W)
        support_feat = STN(support_img)
        support_feat = support_feat / K

        _, C, H, W = support_feat.shape
        support_feat = support_feat.view(B, K, C, H, W)
        support_feat = torch.sum(support_feat, dim=1)

        z1 = ENC(query_feat)
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)
        total_loss = CosLoss(p1,z2, Mean=True)/2 + CosLoss(p2,z1, Mean=True)/2
        total_losses.update(total_loss.item(), query_img.size(0))

        total_loss.backward()

        STN_optimizer.step()
        ENC_optimizer.step()
        PRED_optimizer.step()

    print_log(('Train Epoch: {} Total_Loss: {:.6f}'.format(epoch, total_losses.avg)), log)


def test(models, cur_epoch, fixed_fewshot_list, test_loader, **kwargs):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    support_img = fixed_fewshot_list[cur_epoch]
    augment_support_img = support_img
    # rotate img with small angle
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]

    # torch version
    with torch.no_grad():
        support_feat = STN(augment_support_img.to(device))
    support_feat = torch.mean(support_feat, dim=0, keepdim=True)
    train_outputs['layer1'].append(STN.stn1_output)
    train_outputs['layer2'].append(STN.stn2_output)
    train_outputs['layer3'].append(STN.stn3_output)

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
    train_outputs = [mean, cov]

    # torch version
    query_imgs = []
    gt_list = []
    mask_list = []
    score_map_list = []

    for (query_img, _, mask, y) in test_loader:
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())

        # model prediction
        query_feat = STN(query_img.to(device))
        z1 = ENC(query_feat)
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)

        loss = CosLoss(p1, z2, Mean=False) / 2 + CosLoss(p2, z1, Mean=False) / 2
        loss_reshape = F.interpolate(loss.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                                     align_corners=False).squeeze(0)
        score_map_list.append(loss_reshape.cpu().detach().numpy())

        test_outputs['layer1'].append(STN.stn1_output)
        test_outputs['layer2'].append(STN.stn2_output)
        test_outputs['layer3'].append(STN.stn3_output)

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
    score_map = F.interpolate(dist_list.unsqueeze(1), size=query_img.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()

    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
    return score_map, query_imgs, gt_list, mask_list

def adjust_learning_rate(optimizers, init_lrs, epoch, args):
    """Decay the learning rate based on schedule"""
    for i in range(3):
        cur_lr = init_lrs[i] * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        for param_group in optimizers[i].param_groups:
            param_group['lr'] = cur_lr

def train_patchcore2(models, epoch, train_loader, optimizers, log):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN_optimizer = optimizers[0]
    ENC_optimizer = optimizers[1]
    PRED_optimizer = optimizers[2]

    STN.train()
    ENC.train()
    PRED.train()

    total_losses = AverageMeter()

    features = []
    def hook(module, input, output):
        features.append(output)
    #STN.layer2[-1].register_forward_hook(hook)
    handle = STN.layer3[-1].register_forward_hook(hook)
    
    for (query_img, support_img_list, _) in tqdm(train_loader):
        STN_optimizer.zero_grad()
        ENC_optimizer.zero_grad()
        PRED_optimizer.zero_grad()

        features = []
        query_img = query_img.squeeze(0).to(device)
        _ = STN(query_img)
        z1 = ENC(features[0])

        support_img = support_img_list.squeeze(0).to(device)
        B,K,C,H,W = support_img.shape

        features = []
        support_img = support_img.view(B * K, C, H, W)
        _ = STN(support_img)
        support_feat = features[0] / K

        _, C, H, W = support_feat.shape
        support_feat = support_feat.view(B, K, C, H, W)
        support_feat = torch.sum(support_feat, dim=1)
        
        z2 = ENC(support_feat)
        p1 = PRED(z1)
        p2 = PRED(z2)
        total_loss = CosLoss(p1,z2, Mean=True)/2 + CosLoss(p2,z1, Mean=True)/2
        total_losses.update(total_loss.item(), query_img.size(0))

        total_loss.backward()

        STN_optimizer.step()
        ENC_optimizer.step()
        PRED_optimizer.step()
    handle.remove()

    print_log(('Train Epoch: {} Total_Loss: {:.6f}'.format(epoch, total_losses.avg)), log)


def test_patchcore(models, cur_epoch, fixed_fewshot_list, test_loader, **kwargs):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    feature_pooler = torch.nn.AvgPool2d(3, 1, 1)

    support_img = fixed_fewshot_list[cur_epoch]
    augment_support_img = support_img
    '''
    # rotate img with small angle
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    '''

    with torch.no_grad():
        features = STN(augment_support_img.to(device))
    features = {layer: feature_pooler(feature) for layer, feature in features.items()}
    embedding = generate_embedding(features=features, layers=['layer2','layer3'])
    embedding_vectors = reshape_embedding(embedding)
    sampling_ratio = 0.1
    sampler = KCenterGreedy(embedding=embedding_vectors, sampling_ratio=sampling_ratio)
    memory_bank = sampler.sample_coreset()

    query_imgs = []
    gt_list = []
    mask_list = []
    anomaly_map_list = []
    for (query_img, _, mask, y) in test_loader:
        query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())

        # model prediction
        with torch.no_grad():
            features = STN(query_img.to(device))
        features = {layer: feature_pooler(feature) for layer, feature in features.items()}
        embedding_vectors = generate_embedding(features=features, layers=['layer2','layer3'])
        B, C, H, W = embedding_vectors.size()
        feature_map_shape = embedding_vectors.shape[-2:]
        embedding_vectors = embedding_vectors.view(B * H * W, C)

        #nearest_neighbors
        n_neighbors = 9
        distances = torch.cdist(embedding_vectors, memory_bank, p=2.0)  # euclidean norm
        patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)

        input_size = [224, 224]
        anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        anomaly_map_one, anomaly_score = anomaly_map_generator(
            patch_scores=patch_scores, feature_map_shape=feature_map_shape
        )
        anomaly_map_list.append(anomaly_map_one)
    anomaly_map = torch.cat(anomaly_map_list, 0)
    #import ipdb
    #ipdb.set_trace()
    return anomaly_map.squeeze().cpu(), query_imgs, gt_list, mask_list

def test_patchcore2(models, cur_epoch, fixed_fewshot_list, test_loader, **kwargs):
    STN = models[0]
    ENC = models[1]
    PRED = models[2]

    STN.eval()
    ENC.eval()
    PRED.eval()

    m = torch.nn.AvgPool2d(3, 1, 1)

    support_img = fixed_fewshot_list[cur_epoch]
    augment_support_img = support_img
    
    # rotate img with small angle
    for angle in [-np.pi / 4, -3 * np.pi / 16, -np.pi / 8, -np.pi / 16, np.pi / 16, np.pi / 8, 3 * np.pi / 16,
                  np.pi / 4]:
        rotate_img = rot_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate_img], dim=0)
    # translate img
    for a, b in [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1),
                 (0.1, -0.1)]:
        trans_img = translation_img(support_img, a, b)
        augment_support_img = torch.cat([augment_support_img, trans_img], dim=0)
    # hflip img
    flipped_img = hflip_img(support_img)
    augment_support_img = torch.cat([augment_support_img, flipped_img], dim=0)
    # rgb to grey img
    greyed_img = grey_img(support_img)
    augment_support_img = torch.cat([augment_support_img, greyed_img], dim=0)
    # rotate img in 90 degree
    for angle in [1, 2, 3]:
        rotate90_img = rot90_img(support_img, angle)
        augment_support_img = torch.cat([augment_support_img, rotate90_img], dim=0)
    augment_support_img = augment_support_img[torch.randperm(augment_support_img.size(0))]
    
    features = []
    def hook(module, input, output):
        features.append(output)
    handle1 = STN.layer2[-1].register_forward_hook(hook)
    handle2 = STN.layer3[-1].register_forward_hook(hook)

    with torch.no_grad():
        _ = STN(augment_support_img.to(device))
    embeddings = []
    embedding_list = []
    for feature in features:
        embeddings.append(m(feature))
    handle1.remove()
    handle2.remove()
    embedding = embedding_concat2(embeddings[0], embeddings[1])
    embedding_list.extend(reshape_embedding2(np.array(embedding)))
    total_embeddings = np.array(embedding_list)
    # Random projection
    randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
    randomprojector.fit(total_embeddings)
    # Coreset Subsampling
    coreset_sampling_ratio = 0.1
    selector = kCenterGreedy2(total_embeddings,0,0)
    selected_idx = selector.select_batch(model=randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*coreset_sampling_ratio))
    embedding_coreset = total_embeddings[selected_idx]
    print('initial embedding size : ', total_embeddings.shape)
    print('final embedding size : ', embedding_coreset.shape)
    #faiss
    index = faiss.IndexFlatL2(embedding_coreset.shape[1])
    index.add(embedding_coreset) 
    #faiss.write_index(index, os.path.join(embedding_dir_path,'index.faiss'))

    query_imgs = []
    gt_list = []
    mask_list = []
    anomaly_map_list = []
    pred_list_img_lvl = []
    pred_list_px_lvl = []
    feature_test = []
    def hook_t(module, input, output):
        feature_test.append(output)
    handle1 = STN.layer2[-1].register_forward_hook(hook_t)
    handle2 = STN.layer3[-1].register_forward_hook(hook_t)
    for (query_img, _, mask, y) in test_loader:
        #query_imgs.extend(query_img.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        mask_list.extend(mask.cpu().detach().numpy())

        feature_test = []
        # model prediction
        with torch.no_grad():
            _ = STN(query_img.to(device))
        embeddings_test = []
        for feature in feature_test:
            embeddings_test.append(m(feature))
        embedding_ = embedding_concat2(embeddings_test[0], embeddings_test[1])
        embedding_test = np.array(reshape_embedding2(np.array(embedding_)))
        n_neighbors = 9
        score_patches, _ = index.search(embedding_test , k=n_neighbors)

        N_b = score_patches[np.argmax(score_patches[:,0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        score = w*max(score_patches[:,0]) # Image-level score
        pred_list_img_lvl.append(score)
        
        anomaly_map = score_patches[:,0].reshape((28,28))
        anomaly_map_resized = cv2.resize(anomaly_map, (224,224))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
    handle1.remove()
    handle2.remove()
    #anomaly_map = np.stack(anomaly_map_list, 0)
    #import ipdb
    #ipdb.set_trace()
    return pred_list_px_lvl, pred_list_img_lvl, gt_list, mask_list

if __name__ == '__main__':
    main()
