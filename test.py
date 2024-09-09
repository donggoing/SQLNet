#!/usr/bin/env python
# coding: utf-8

"""
Test code written by Viresh Ranjan

Last modified by: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
Date: 2021/04/19
"""


import os
from datetime import datetime
import torch
import argparse
import numpy as np
from tqdm import tqdm
from os.path import exists
import torch.distributed as dist

from models.backbone import Resnet50FPN
from models.p2pnet import build
from models.feature_fuser import Fusion
from util.utils import MAPS, MAPS_dim, Scales, extract_features, vis
from data.build_dataset import build_dataset


def eval(test_data, vis_dir=None):
    cnt = torch.tensor(0).cuda()
    SAE = torch.tensor(0.).cuda() # sum of absolute errors
    SSE = torch.tensor(0.).cuda() # sum of square errors
    d_SAE = torch.tensor(0.).cuda() # sum of density absolute errors
    d_SSE = torch.tensor(0.).cuda() # sum of density square errors

    if args.local_rank == 0:
        pbar = tqdm(test_data, dynamic_ncols=True)
    else:
        pbar = test_data

    use_time = 0

    for i, (images, dens, rects, dots, im_names, center_indexes) in enumerate(pbar):
        
        images = images.cuda()
        dots = [dot.cuda() for dot in dots]
        rects = [rect.cuda() for rect in rects]
        with torch.no_grad():
            image_features, exampler_features, h_scales, w_scales = extract_features(resnet50_conv, images, rects, MAPS, Scales, return_all_feat=False)
            
            image_features = feature_fuser(image_features, exampler_features, h_scales, w_scales)
            outputs = p2p(images.shape[2:],image_features)
            
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

            # 0.5 is used by default
            threshold = 0.5
            predict_cnt = int((outputs_scores > threshold).sum())

            outputs_points = outputs['pred_points'][0]
            outputs_sizes = outputs['pred_sizes'][0]
            points_ind = torch.where(outputs_scores > threshold)
            points = outputs_points[points_ind].detach().cpu().numpy()
            scores = outputs_scores[points_ind].detach().cpu().numpy()
            sizes = outputs_sizes[points_ind].detach().cpu().numpy()
    

        gt_cnt = len(dots[0])
        cnt += images.shape[0]
        err = abs(gt_cnt - predict_cnt)
        
        SAE += err
        SSE += err**2
        
        cur_cnt = cnt.clone().detach()
        cur_SAE = SAE.clone().detach()
        cur_SSE = SSE.clone().detach()
            
        stat_str = '{:<8}: {:6d} predicted: {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}, density MAE: {:5.2f}, density RMSE: {:5.2f}'.format(im_names[0], gt_cnt, predict_cnt, abs(predict_cnt - gt_cnt), cur_SAE/cur_cnt, (cur_SSE/cur_cnt)**0.5, d_SAE/cur_cnt, (d_SSE/cur_cnt)**0.5)
        pbar.set_description(stat_str)
        with open(args.stats_file, 'a') as f:
            f.write(stat_str + '\n')
            
        if args.vis and vis_dir is not None:
            rslt_file = "{}/{}_{}_{}_{}.jpg".format(vis_dir, os.path.splitext(im_names[0])[0], gt_cnt, round(predict_cnt), round(err))
            vis(images.detach().cpu(), points, rects[0].cpu(), rslt_file, ds=2, dots=dots[0].cpu(), pred_sizes=sizes, pred_score=scores)
            
        print('Avg used time: {:.4f} s, FPS: {:.4f}'.format(use_time/cnt, cnt/use_time))

    return (cur_SAE/cur_cnt).item(), ((cur_SSE/cur_cnt)**0.5).item()

parser = argparse.ArgumentParser(description="Few Shot Counting Evaluation code")
parser.add_argument("-dp", "--data_path", type=str, default='/path/to/FSC147_384_V2/', help="Path to the FSC147 dataset")
parser.add_argument("-ts", "--test_split", type=str, default='val', choices=["train", "test", "val", "test_coco", "val_coco"], help="what data split to evaluate on")
parser.add_argument("-m",  "--model_path", type=str, default="./weights/SQLNet.pth", help="path to trained model")
parser.add_argument("-en",  "--exp_name", type=str, default="test", help="path to trained model")
parser.add_argument("-g",  "--gpu", type=str, default='1', help="GPU id. Default 0 for the first GPU. Use -1 for CPU.")
parser.add_argument("-v",  "--vis", action='store_true', help="If specified, visualize the results")
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')

args = parser.parse_args()


device_ids=[int(g) for g in args.gpu.split(',')]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.exp_name != "":
    args.output_dir = os.path.join('exps', args.exp_name)
    model_path = os.path.join(args.output_dir, 'FamNet.pth')
    if not exists(model_path):
        model_path = args.model_path
        print("Exp's model does not exist. Using {}".format(args.model_path))
    args.model_path = model_path
    args.stats_file = os.path.join(args.output_dir, 'stats_{}_{}.txt'.format(args.test_split, datetime.now().strftime("%Y%m%d-%H%M%S")))
    if args.vis:
        args.vis_dir = os.path.join(args.output_dir, 'vis_{}'.format(args.test_split))
        if not exists(args.vis_dir):
            os.makedirs(args.vis_dir)
    
with open(args.stats_file, 'a') as f:
    f.write("{}\n".format(args))

resnet50_conv = Resnet50FPN(use_moco_pretrained=True)
resnet50_conv.cuda()
resnet50_conv.eval()

model_params = torch.load(args.model_path)


layer_dim = [MAPS_dim[m] for m in MAPS]
feature_fuser = Fusion(layer_dim=layer_dim, N=2, tr_token_dim=1280, d_ff=1024, d_kvfeature=1024, d_qfeature=1024, h=8, dropout=0.1)
feature_fuser.load_state_dict(model_params['feature_fuser'])
feature_fuser.cuda()
feature_fuser.eval()

args.row = 2
args.line = 2
p2p = build(args, False)
p2p.load_state_dict(model_params['p2p'], strict=False)
p2p.cuda()
p2p.eval()

test_data = build_dataset(args.data_path, args.test_split, distributed=dist.is_initialized())

eval(test_data, args.vis_dir)