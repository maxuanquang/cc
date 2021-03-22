# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.

import argparse
import os
from tqdm import tqdm
import numpy as np
from path import Path
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import models
import custom_transforms
from inverse_warp import pose2flow
from datasets.validation_flow import ValidationMask
from logger import AverageMeter
from PIL import Image
from torchvision.transforms import ToPILImage
from flowutils.flowlib import flow_to_image
from utils import tensor2array
from loss_functions import compute_all_epes
from scipy.ndimage.interpolation import zoom

parser = argparse.ArgumentParser(description='Test IOU of Mask predictions',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--kitti-dir', dest='kitti_dir', type=str, default='/ps/project/datasets/AllFlowData/kitti/kitti2015',
                    help='Path to kitti2015 scene flow dataset for optical flow validation')
parser.add_argument('--dispnet', dest='dispnet', type=str, default='DispResNet6', choices=['DispNetS', 'DispResNet6', 'DispNetS6'],
                    help='depth network architecture.')
parser.add_argument('--posenet', dest='posenet', type=str, default='PoseNetB6', choices=['PoseNet6','PoseNetB6', 'PoseExpNet5', 'PoseExpNet6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--masknet', dest='masknet', type=str, default='MaskNet6', choices=['MaskResNet6', 'MaskNet6', 'PoseExpNet5', 'PoseExpNet6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--flownet', dest='flownet', type=str, default='Back2Future', choices=['FlowNetS', 'Back2Future', 'FlowNetC5','FlowNetC6', 'SpyNet'],
                    help='flow network architecture.')

parser.add_argument('--THRESH', dest='THRESH', type=float, default=0.94, help='THRESH')

parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained posenet model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH', help='path to pre-trained flownet model')
parser.add_argument('--pretrained-mask', dest='pretrained_mask', default=None, metavar='PATH', help='path to pre-trained masknet model')

parser.add_argument('--nlevels', dest='nlevels', type=int, default=6, help='number of levels in multiscale. Options: 4|5')
parser.add_argument('--dataset', dest='dataset', default='kitti2015', help='path to pre-trained Flow net model')
parser.add_argument('--output-dir', dest='output_dir', type=str, default=None, help='path to output directory')


def validate_mask_with_gt(disp_net,pose_net,mask_net,flow_net,val_loader):

    def mask_error(mot_gt, seg_gt, pred):
        max_label = 2
        tp = np.zeros((max_label))
        fp = np.zeros((max_label))
        fn = np.zeros((max_label))

        mot_gt[mot_gt != 0] = 1
        mov_car_gt = mot_gt
        mov_car_gt[seg_gt != 26] = 255
        mot_gt = mov_car_gt
        r_shape = [float(i) for i in list(pred.shape)]
        g_shape = [float(i) for i in list(mot_gt.shape)]
        pred = zoom(pred, (g_shape[0] / r_shape[0],
                        g_shape[1] / r_shape[1]), order  = 0)

        if len(pred.shape) == 2:
            mask = pred
            umask = np.zeros((2, mask.shape[0], mask.shape[1]))
            umask[0, :, :] = mask
            umask[1, :, :] = 1. - mask
            pred = umask

        pred = pred.argmax(axis=0)
        if (np.max(pred) > (max_label - 1) and np.max(pred)!=255):
            print('Result has invalid labels: ', np.max(pred))
        else:
            # For each class
            for class_id in range(0, max_label):
                class_gt = np.equal(mot_gt, class_id)
                class_result = np.equal(pred, class_id)
                class_result[np.equal(mot_gt, 255)] = 0
                tp[class_id] = tp[class_id] +\
                    np.count_nonzero(class_gt & class_result)
                fp[class_id] = fp[class_id] +\
                    np.count_nonzero(class_result & ~class_gt)
                fn[class_id] = fn[class_id] +\
                    np.count_nonzero(~class_result & class_gt)

        return [tp[0], fp[0], fn[0], tp[1], fp[1], fn[1]]

    nlevels = 6
    THRESH = 0.94

    error_names = ['tp_0', 'fp_0', 'fn_0', 'tp_1', 'fp_1', 'fn_1']
    errors = AverageMeter(i=len(error_names))
    errors_census = AverageMeter(i=len(error_names))
    errors_bare = AverageMeter(i=len(error_names))

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt, semantic_map_gt) in enumerate(tqdm(val_loader)):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        ref_imgs_var = [Variable(img.cuda(), volatile=True) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda(), volatile=True)
        intrinsics_inv_var = Variable(intrinsics_inv.cuda(), volatile=True)

        flow_gt_var = Variable(flow_gt.cuda(), volatile=True)
        obj_map_gt_var = Variable(obj_map_gt.cuda(), volatile=True)

        disp = disp_net(tgt_img_var)
        depth = 1/disp
        pose = pose_net(tgt_img_var, ref_imgs_var)
        explainability_mask = mask_net(tgt_img_var, ref_imgs_var)

        flow_fwd, flow_bwd, _ = flow_net(tgt_img_var, ref_imgs_var[1:3])

        flow_cam = pose2flow(depth.squeeze(1), pose[:,2], intrinsics_var, intrinsics_inv_var)

        rigidity_mask = 1 - (1-explainability_mask[:,1])*(1-explainability_mask[:,2]).unsqueeze(1) > 0.5
        rigidity_mask_census_soft = (flow_cam - flow_fwd).pow(2).sum(dim=1).unsqueeze(1).sqrt()#.normalize()
        rigidity_mask_census_soft = 1 - rigidity_mask_census_soft/rigidity_mask_census_soft.max()
        rigidity_mask_census = rigidity_mask_census_soft > THRESH

        rigidity_mask_combined = 1 - (1-rigidity_mask.type_as(explainability_mask))*(1-rigidity_mask_census.type_as(explainability_mask))

        flow_fwd_non_rigid = (1- rigidity_mask_combined).type_as(flow_fwd).expand_as(flow_fwd) * flow_fwd
        flow_fwd_rigid = rigidity_mask_combined.type_as(flow_fwd).expand_as(flow_fwd) * flow_cam
        total_flow = flow_fwd_rigid + flow_fwd_non_rigid

        obj_map_gt_var_expanded = obj_map_gt_var.unsqueeze(1).type_as(flow_fwd)

        tgt_img_np = tgt_img[0].numpy()
        rigidity_mask_combined_np = rigidity_mask_combined.cpu().data[0].numpy()
        rigidity_mask_census_np = rigidity_mask_census.cpu().data[0].numpy()
        rigidity_mask_bare_np = rigidity_mask.cpu().data[0].numpy()

        gt_mask_np = obj_map_gt[0].numpy()
        semantic_map_np = semantic_map_gt[0].numpy()

        _errors = mask_error(gt_mask_np, semantic_map_np, rigidity_mask_combined_np[0])
        _errors_census = mask_error(gt_mask_np, semantic_map_np, rigidity_mask_census_np[0])
        _errors_bare = mask_error(gt_mask_np, semantic_map_np, rigidity_mask_bare_np[0])

        errors.update(_errors)
        errors_census.update(_errors_census)
        errors_bare.update(_errors_bare)

    bg_iou = errors.sum[0] / (errors.sum[0] + errors.sum[1] + errors.sum[2]  )
    fg_iou = errors.sum[3] / (errors.sum[3] + errors.sum[4] + errors.sum[5]  )
    avg_iou = (bg_iou + fg_iou)/2

    bg_iou_census = errors_census.sum[0] / (errors_census.sum[0] + errors_census.sum[1] + errors_census.sum[2]  )
    fg_iou_census = errors_census.sum[3] / (errors_census.sum[3] + errors_census.sum[4] + errors_census.sum[5]  )
    avg_iou_census = (bg_iou_census + fg_iou_census)/2

    bg_iou_bare = errors_bare.sum[0] / (errors_bare.sum[0] + errors_bare.sum[1] + errors_bare.sum[2]  )
    fg_iou_bare = errors_bare.sum[3] / (errors_bare.sum[3] + errors_bare.sum[4] + errors_bare.sum[5]  )
    avg_iou_bare = (bg_iou_bare + fg_iou_bare)/2

    error_names = ['iou_full', 'iou_census', 'iou_bare']
    errors_return = (avg_iou, avg_iou_census, avg_iou_bare)
    # print("Results Full Model")
    # print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    # print("Errors \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou, bg_iou, fg_iou))

    # print("Results Census only")
    # print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    # print("Errors \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou_census, bg_iou_census, fg_iou_census))

    # print("Results Bare")
    # print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    # print("Errors \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou_bare, bg_iou_bare, fg_iou_bare))
    return errors_return, error_names
