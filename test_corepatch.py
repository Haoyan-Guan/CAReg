import os
import random
import argparse
import time
import torch
import numpy as np
from torch.optim import optimizer
import torch.nn.functional as F
from tqdm import tqdm
from datasets.mvtec import FSAD_Dataset_train, FSAD_Dataset_test
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from models.siamese import Encoder, Predictor
from models.stn import stn_net
from losses.norm_loss import CosLoss
from utils.funcs import embedding_concat, mahalanobis_torch, rot_img, translation_img, hflip_img, rot90_img, grey_img
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

from PIL import Image
from sklearn.random_projection import SparseRandomProjection
import faiss
from utils.utils import KCenterGreedy, AnomalyMapGenerator, kCenterGreedy2
from torchvision import transforms
import cv2
from sklearn.metrics import precision_recall_curve

import matplotlib
import matplotlib.pyplot as plt
from skimage import morphology, measure
from skimage.segmentation import mark_boundaries

def denormalization(x):
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    #x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    x = (((np.transpose(x, (1, 2, 0))) * std + mean) * 255.).astype(np.uint8)
    return x

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

def main():
    parser = argparse.ArgumentParser(description='RegAD on MVtec')
    parser.add_argument('--obj', type=str, default='bracket_brown')
    #parser.add_argument('--data_type', type=str, default='mvtec')
    #parser.add_argument('--data_path', type=str, default='./MVTec/')
    parser.add_argument('--data_type', type=str, default='MPDD')
    parser.add_argument('--data_path', type=str, default='./MPDD/')
    parser.add_argument('--epochs', type=int, default=50, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate in SGD')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
    parser.add_argument('--seed', type=int, default=668, help='manual seed')
    parser.add_argument('--shot', type=int, default=2, help='shot count')
    parser.add_argument('--inferences', type=int, default=10, help='number of rounds per inference')
    parser.add_argument('--stn_mode', type=str, default='rotation_scale', help='[affine, translation, rotation, scale, shear, rotation_scale, translation_scale, rotation_translation, rotation_translation_scale]')
    parser.add_argument('--coreset_sampling_ratio', type=float, default=0.1, help='memory bank rate')
    args = parser.parse_args()

    args.input_channel = 3
    if args.seed is None:
        args.seed = random.randint(1, 10000)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    args.prefix = time_file_str()

    STN = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True).to(device)
    ENC = Encoder().to(device)
    PRED = Predictor().to(device)

    # load models
    #CKPT_name = f'./logs_mvtec/rotation_scale/{args.shot}/{args.obj}/{args.obj}_{args.shot}_rotation_scale_model.pt'
    CKPT_name = f'./logs_mpdd/affine/{args.shot}/{args.obj}/{args.obj}_{args.shot}_affine_model.pt'
    model_CKPT = torch.load(CKPT_name)
    STN.load_state_dict(model_CKPT['STN'])
    ENC.load_state_dict(model_CKPT['ENC'])
    PRED.load_state_dict(model_CKPT['PRED'])
    models = [STN, ENC, PRED]

    #print('Loading Datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    #test_dataset = FSAD_Dataset_test(args.data_path, class_name=args.obj, is_train=False, resize=args.img_size, shot=args.shot)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    print('Loading Fixed Support Set')
    fixed_fewshot_list = torch.load(f'./support_set/{args.obj}/{args.shot}_{args.inferences}.pt')

    print('Start Testing:')
    image_auc_list = []
    pixel_auc_list = []

    args.save_dir = './vis_diversity_MPDD_0.25/' + args.obj + '/'
    main_dir = '../../Dataset/MPDD/'
    #query_image_dir = main_dir + 'pill/test/color/007.png'
    #query_mask_dir = main_dir + 'pill/ground_truth/color/007_mask.png'

    #query_image_dir = main_dir + 'tile/test/rough/014.png'
    #query_mask_dir = main_dir + 'tile/ground_truth/rough/014_mask.png'

    #query_image_dir = main_dir + 'wood/test/color/003.png'
    #query_mask_dir = main_dir + 'wood/ground_truth/color/003_mask.png'

    #main_dir = '../../Dataset/MPDD/'
    #query_image_dir = main_dir + 'connector/test/parts_mismatch/011.png'
    #query_mask_dir = main_dir + 'connector/ground_truth/parts_mismatch/011_mask.png'

    #anomaly_type = 'hole008'
    #query_image_dir = main_dir + 'wood/test/combined/007.png'
    #query_mask_dir = main_dir + 'wood/ground_truth/combined/007_mask.png'
    #query_image_dir = main_dir + 'wood/test/color/001.png'
    #query_mask_dir = main_dir + 'wood/ground_truth/color/001_mask.png'
    #query_image_dir = main_dir + 'wood/test/hole/008.png'
    #query_mask_dir = main_dir + 'wood/ground_truth/hole/008_mask.png'
    #query_image_dir = main_dir + 'wood/test/liquid/004.png'
    #query_mask_dir = main_dir + 'wood/ground_truth/liquid/004_mask.png'
    #query_image_dir = main_dir + 'wood/test/scratch/014.png'
    #query_mask_dir = main_dir + 'wood/ground_truth/scratch/014_mask.png'

    #anomaly_type = 'misplaced002'
    #query_image_dir = main_dir + 'transistor/test/bent_lead/005.png'
    #query_mask_dir = main_dir + 'transistor/ground_truth/bent_lead/005_mask.png'
    #query_image_dir = main_dir + 'transistor/test/cut_lead/007.png'
    #query_mask_dir = main_dir + 'transistor/ground_truth/cut_lead/007_mask.png'
    #query_image_dir = main_dir + 'transistor/test/damaged_case/009.png'
    #query_mask_dir = main_dir + 'transistor/ground_truth/damaged_case/009_mask.png'
    #query_image_dir = main_dir + 'transistor/test/misplaced/002.png'
    #query_mask_dir = main_dir + 'transistor/ground_truth/misplaced/002_mask.png'

    anomaly_type = 'parts024'
    #query_image_dir = main_dir + 'bracket_brown/test/bend_and_parts_mismatch/009.png'
    #query_mask_dir = main_dir + 'bracket_brown/ground_truth/bend_and_parts_mismatch/009_mask.png'
    query_image_dir = main_dir + 'bracket_brown/test/parts_mismatch/024.png'
    query_mask_dir = main_dir + 'bracket_brown/ground_truth/parts_mismatch/024_mask.png'

    query_img = Image.open(query_image_dir).convert('RGB')
    mask = Image.open(query_mask_dir)
    transform_x = transforms.Compose([
            transforms.Resize(args.img_size, Image.ANTIALIAS),
            transforms.ToTensor(),
        ])
    transform_mask = transforms.Compose(
        [transforms.Resize(args.img_size, Image.NEAREST),
            transforms.ToTensor()])
    query_img = transform_x(query_img)
    mask = transform_mask(mask)
    #for inference_round in range(args.inferences):
    #print('Round {}:'.format(inference_round))
    #gt_list_px_lvl, gt_list_img_lvl, gt_list, gt_mask_list = test(args, models, inference_round, fixed_fewshot_list, test_loader, **kwargs)
    for inference_round in range(args.inferences):
        print('Round {}:'.format(inference_round))
        gt_list_px_lvl, gt_list_img_lvl, gt_list, gt_mask_list = test_oneimage(args, models, inference_round, fixed_fewshot_list, query_img,mask, **kwargs)

        #img_roc_auc = roc_auc_score(gt_list, gt_list_img_lvl)
        #image_auc_list.append(img_roc_auc)

        scores = np.asarray(gt_list_px_lvl)
        # Normalization
        max_anomaly_score = scores.max()
        min_anomaly_score = scores.min()
        scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
        #gt_mask = np.asarray(gt_mask_list)
        gt_mask = gt_mask_list[0].numpy()
        gt_mask = (gt_mask > 0.5).astype(np.int_)
        #import ipdb
        #ipdb.set_trace()
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        pixel_auc_list.append(per_pixel_rocauc)

        precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), scores.flatten())
        a = 2 * precision * recall
        b = precision + recall
        f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        threshold = thresholds[np.argmax(f1)]
        #save_dir = args.save_dir + 'pictures_{}_{:.4f}'.format(inference_round,threshold)
        save_dir = args.save_dir + anomaly_type +'_{}_{:.4f}'.format(inference_round,threshold)
        #save_dir = args.save_dir + 'pictures_{}'.format(inference_round)
        os.makedirs(save_dir, exist_ok=True)
        plot_fig_new(args, query_img.numpy(), scores.reshape(224,224), gt_mask_list[0].numpy(), threshold, save_dir)

    #image_auc_list = np.array(image_auc_list)
    pixel_auc_list = np.array(pixel_auc_list)
    #mean_img_auc = np.mean(image_auc_list, axis = 0)
    mean_pixel_auc = np.mean(pixel_auc_list, axis = 0)
    #print('Img-level AUC:',mean_img_auc)
    print('Pixel-level AUC:', mean_pixel_auc)

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
    #for i in range(num):
    img = test_img
    #import ipdb
    #ipdb.set_trace()
    img = denormalization(img)
    #recon_img = recon_imgs[i]
    #recon_img = denormalization(recon_img)
    #gt = gts.transpose(1, 2, 0).squeeze()
    gt = np.transpose(gts, (1, 2, 0)).squeeze()
    heat_map = scores * 255
    mask = scores
    mask[mask > threshold] = 1
    mask[mask <= threshold] = 0
    kernel = morphology.disk(4)
    #import ipdb
    #ipdb.set_trace()
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

    fig_img.savefig(os.path.join(save_dir, 'all', args.obj), dpi=100)
    plt.close()
    plt.imshow(img, cmap='gray', interpolation='none')
    plt.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
    plt.savefig(os.path.join(save_dir, 'heat', args.obj))
    plt.close()
    plt.imshow(mask, cmap='gray')
    plt.savefig(os.path.join(save_dir, 'res', args.obj))
    plt.close()
    plt.imshow(vis_img)
    plt.savefig(os.path.join(save_dir, 'pre', args.obj))
    plt.close()

def test_oneimage(args, models, cur_epoch, fixed_fewshot_list, query_img,mask, **kwargs):
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
    coreset_sampling_ratio = args.coreset_sampling_ratio
    #coreset_sampling_ratio = 0.01
    selector = kCenterGreedy2(total_embeddings,0,0)
    selected_idx = selector.select_batch(model=randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*coreset_sampling_ratio))
    embedding_coreset = total_embeddings[selected_idx]
    print('initial embedding size : ', total_embeddings.shape)
    print('final embedding size : ', embedding_coreset.shape)
    #faiss
    index = faiss.IndexFlatL2(embedding_coreset.shape[1])
    index.add(embedding_coreset) 
    #faiss.write_index(index, os.path.join(embedding_dir_path,'index.faiss'))

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

    #for (query_img, _, mask, y) in test_loader:
    y = np.array(1)
    #query_imgs.extend(query_img.cpu().detach().numpy())
    gt_list.append(y)
    mask_list.append(mask)

    feature_test = []
    # model prediction
    with torch.no_grad():
        _ = STN(query_img.unsqueeze(0).to(device))
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
