# Author: Anurag Ranjan
# Copyright (c) 2019, Anurag Ranjan
# All rights reserved.
# based on github.com/ClementPinard/SfMLearner-Pytorch

import argparse
import time
import csv
import datetime
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.utils.data
import custom_transforms
import models
from utils import tensor2array, save_checkpoint
from inverse_warp import inverse_warp, pose2flow, flow2oob, flow_warp
from loss_functions import compute_joint_mask_for_depth
from loss_functions import consensus_exp_masks, consensus_depth_flow_mask, explainability_loss, gaussian_explainability_loss, smooth_loss, edge_aware_smoothness_loss
from loss_functions import photometric_reconstruction_loss, photometric_flow_loss
from loss_functions import compute_errors, compute_epe, compute_all_epes, flow_diff, spatial_normalize
from logger import TermLogger, AverageMeter
from path import Path
from itertools import chain
from tensorboardX import SummaryWriter
from flowutils.flowlib import flow_to_image
from inverse_warp import pose_vec2mat
from tqdm import tqdm

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

epsilon = 1e-8

parser = argparse.ArgumentParser(description='Competitive Collaboration training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--kitti-dir', dest='kitti_dir', type=str, default='/content/drive/MyDrive/Dự án/KITTI Dataset/Stereo/Stereo 2015',
                    help='Path to kitti2015 scene flow dataset for optical flow validation')
parser.add_argument('--kitti-odometry-dir', dest='kitti_odometry_dir', type=str, default='/content/drive/MyDrive/Dự án/KITTI Dataset/Odometry/dataset',
                    help='Path to kitti odometry dataset for pose validation')
parser.add_argument('--DEBUG', action='store_true', help='DEBUG Mode')
parser.add_argument('--name', dest='name', type=str, default='demo', required=True,
                    help='name of the experiment, checpoints are stored in checpoints/name')
parser.add_argument('--dataset-format', default='sequential', metavar='STR',
                    help='dataset format, stacked: stacked frames (from original TensorFlow code) \
                    sequential: sequential folders (easier to convert to with a non KITTI/Cityscape dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=5)
parser.add_argument('--rotation-mode', type=str, choices=['euler', 'quat'], default='euler',
                    help='rotation mode for PoseExpnet : euler (yaw,pitch,roll) or quaternion (last 3 coefficients)')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')

parser.add_argument('--with-depth-gt', action='store_true', help='use ground truth for depth validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--with-flow-gt', action='store_true', help='use ground truth for flow validation. \
                    see data/validation_flow for an example')
parser.add_argument('--with-pose-gt', action='store_true', help='use ground truth for depth validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
parser.add_argument('--with-mask-gt', action='store_true', help='use ground truth for flow validation. \
                    see data/validation_flow for an example')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--smoothness-type', dest='smoothness_type', type=str, default='regular', choices=['edgeaware', 'regular'],
                    help='Compute mean-std locally or globally')
parser.add_argument('--data-normalization', dest='data_normalization', type=str, default='global', choices=['local', 'global'],
                    help='Compute mean-std locally or globally')
parser.add_argument('--nlevels', dest='nlevels', type=int, default=6,
                    help='number of levels in multiscale. Options: 6')

parser.add_argument('--dispnet', dest='dispnet', type=str, default='DispNetS', choices=['DispNetS', 'DispNetS6', 'DispResNetS6', 'DispResNet6'],
                    help='depth network architecture.')
parser.add_argument('--posenet', dest='posenet', type=str, default='PoseNet', choices=['PoseNet6','PoseNetB6', 'PoseExpNet'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--masknet', dest='masknet', type=str, default='MaskNet', choices=['MaskResNet6', 'MaskNet6'],
                    help='pose and explainabity mask network architecture. ')
parser.add_argument('--flownet', dest='flownet', type=str, default='FlowNetS', choices=['Back2Future', 'FlowNetC6'],
                    help='flow network architecture. Options: FlowNetC6 | Back2Future')

parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH',
                    help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-mask', dest='pretrained_mask', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH',
                    help='path to pre-trained Exp Pose net model')
parser.add_argument('--pretrained-flow', dest='pretrained_flow', default=None, metavar='PATH',
                    help='path to pre-trained Flow net model')
parser.add_argument('--pretrained-flow-lua', dest='pretrained_flow_lua', default=None, metavar='PATH',
                    help='path to pre-trained Flow net model converted from Lua model')

parser.add_argument('--spatial-normalize', dest='spatial_normalize', action='store_true', help='spatially normalize depth maps')
parser.add_argument('--robust', dest='robust', action='store_true', help='train using robust losses')
parser.add_argument('--no-non-rigid-mask', dest='no_non_rigid_mask', action='store_true', help='will not use mask on loss of non-rigid flow')
parser.add_argument('--joint-mask-for-depth', dest='joint_mask_for_depth', action='store_true', help='use joint mask from masknet and consensus mask for depth training')
parser.add_argument('--occ-mask-at-scale', dest='occ_mask_at_scale', action='store_true', help='use occlusion mask at scale for flow photometric loss')

parser.add_argument('--fix-masknet', dest='fix_masknet', action='store_true', help='do not train posenet')
parser.add_argument('--fix-posenet', dest='fix_posenet', action='store_true', help='do not train posenet')
parser.add_argument('--fix-flownet', dest='fix_flownet', action='store_true', help='do not train flownet')
parser.add_argument('--fix-dispnet', dest='fix_dispnet', action='store_true', help='do not train dispnet')

parser.add_argument('--alternating', dest='alternating', action='store_true', help='minimize only one network at a time')
parser.add_argument('--clamp-masks', dest='clamp_masks', action='store_true', help='threshold masks for training')
parser.add_argument('--fix-posemasknet', dest='fix_posemasknet', action='store_true', help='fix pose and masknet')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH',
                    help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH',
                    help='csv where to save per-gradient descent train stats')
parser.add_argument('-qch', '--qch', type=float, help='q value for charbonneir', metavar='W', default=0.5)
parser.add_argument('-wrig', '--wrig', type=float, help='consensus imbalance weight', metavar='W', default=1.0)
parser.add_argument('-wbce', '--wbce', type=float, help='weight for binary cross entropy loss', metavar='W', default=0.5)
parser.add_argument('-wssim', '--wssim', type=float, help='weight for ssim loss', metavar='W', default=0.0)
parser.add_argument('-pc', '--cam-photo-loss-weight', type=float, help='weight for camera photometric loss for rigid pixels', metavar='W', default=1)
parser.add_argument('-pf', '--flow-photo-loss-weight', type=float, help='weight for photometric loss for non rigid optical flow', metavar='W', default=1)
parser.add_argument('-m', '--mask-loss-weight', type=float, help='weight for explainabilty mask loss', metavar='W', default=0)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--consensus-loss-weight', type=float, help='weight for mask consistancy', metavar='W', default=0.1)
parser.add_argument('--THRESH', '--THRESH', type=float, help='threshold for masks', metavar='W', default=0.01)
parser.add_argument('--lambda-oob', type=float, help='weight on the out of bound pixels', default=0)
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs and warped imgs at validation step')
parser.add_argument('--log-terminal', action='store_true', help='will display progressbar at terminal')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('-f', '--training-output-freq', type=int, help='frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output',
                    metavar='N', default=0)

parser.add_argument('--break-training-if-saturate', action='store_true', help='break training process if validation error saturates')
parser.add_argument('--break-training-value-threshold', type=float, default=0.03, help='if changes in validation error less than value-threshold then it will be considered saturated')
parser.add_argument('--break-training-count-threshold', type=int, default=2, help='if validation error saturates more than count-threshold then break training process')

best_error = -1000
n_iter = 0
start_epoch = 0
saturation_count = 0


def main():
    global args, best_error, n_iter, start_epoch, saturation_count, epsilon
    args = parser.parse_args()
    if args.dataset_format == 'stacked':
        from datasets.stacked_sequence_folders import SequenceFolder
    elif args.dataset_format == 'sequential':
        from datasets.sequence_folders import SequenceFolder
    save_path = Path(args.name)
    args.save_path = '/content/drive/MyDrive/VinAI/Motion segmentation/checkpoints'/save_path #/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    torch.manual_seed(args.seed)
    if args.alternating:
        args.alternating_flags = np.array([False,False,True])

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # Data loading code
    flow_loader_h, flow_loader_w = 256, 832

    if args.data_normalization =='global':
        normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])
    elif args.data_normalization =='local':
        normalize = custom_transforms.NormalizeLocally()

    if args.fix_flownet:
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])
    else:
        train_transform = custom_transforms.Compose([
            custom_transforms.RandomRotate(),
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    valid_flow_transform = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                            custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_depth_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data.replace('cityscapes', 'kitti'),
            transform=valid_transform
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
        )

    if args.with_flow_gt:
        from datasets.validation_flow import ValidationFlow
        val_flow_set = ValidationFlow(root=args.kitti_dir,
                                        sequence_length=args.sequence_length, transform=valid_flow_transform)

    if args.with_mask_gt:
        normalize_ = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])
        flow_loader_h, flow_loader_w = 256, 832
        valid_flow_transform_ = custom_transforms.Compose([custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
                                custom_transforms.ArrayToTensor(), normalize_])
        val_flow_set_ = ValidationMask(root=args.kitti_dir, N=100,
                                    sequence_length=5, transform=valid_flow_transform_)

        val_mask_loader = torch.utils.data.DataLoader(val_flow_set_, batch_size=1, shuffle=False,
            num_workers=2, pin_memory=True, drop_last=True)

    if args.with_pose_gt:
        from kitti_eval.pose_evaluation_utils import test_framework_KITTI_for_pose_validation as test_framework
        framework = test_framework(Path(args.kitti_odometry_dir), ['09'], args.sequence_length)

    if args.DEBUG:
        train_set.__len__ = 32
        train_set.samples = train_set.samples[:32]

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.with_flow_gt:
        val_flow_loader = torch.utils.data.DataLoader(val_flow_set, batch_size=1,               # batch size is 1 since images in kitti have different sizes
                        shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")

    disp_net = getattr(models, args.dispnet)().cuda()
    output_exp = True #args.mask_loss_weight > 0
    if not output_exp:
        print("=> no mask loss, PoseExpnet will only output pose")
    pose_net = getattr(models, args.posenet)(nb_ref_imgs=args.sequence_length - 1).cuda()
    mask_net = getattr(models, args.masknet)(nb_ref_imgs=args.sequence_length - 1, output_exp=True).cuda()

    if args.flownet=='SpyNet':
        flow_net = getattr(models, args.flownet)(nlevels=args.nlevels, pre_normalization=normalize).cuda()
    else:
        flow_net = getattr(models, args.flownet)(nlevels=args.nlevels).cuda()

    if args.pretrained_pose:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'])
    else:
        pose_net.init_weights()

    if args.pretrained_mask:
        print("=> using pre-trained weights for explainabilty and pose net")
        weights = torch.load(args.pretrained_mask)
        mask_net.load_state_dict(weights['state_dict'])
    else:
        mask_net.init_weights()

    # import ipdb; ipdb.set_trace()
    if args.pretrained_disp:
        print("=> using pre-trained weights from {}".format(args.pretrained_disp))
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'])
    else:
        disp_net.init_weights()

    if args.pretrained_flow:
        print("=> using pre-trained weights for FlowNet")
        weights = torch.load(args.pretrained_flow)
        flow_net.load_state_dict(weights['state_dict'])
    elif args.pretrained_flow_lua:
        print("=> using pre-trained weights for FlowNet")
        weights = torch.load(args.pretrained_flow_lua)
        flow_net.load_state_dict(weights)
    else:
        flow_net.init_weights()

    if args.resume:
        print("=> resuming from checkpoint")
        dispnet_weights = torch.load(args.save_path/'dispnet_checkpoint.pth.tar')
        posenet_weights = torch.load(args.save_path/'posenet_checkpoint.pth.tar')
        masknet_weights = torch.load(args.save_path/'masknet_checkpoint.pth.tar')
        flownet_weights = torch.load(args.save_path/'flownet_checkpoint.pth.tar')
        disp_net.load_state_dict(dispnet_weights['state_dict'])
        pose_net.load_state_dict(posenet_weights['state_dict'])
        flow_net.load_state_dict(flownet_weights['state_dict'])
        mask_net.load_state_dict(masknet_weights['state_dict'])
        with open(os.path.join(args.save_path,'n_iter.txt'),'r') as f:
            n_iter = int(f.readline())
        with open(os.path.join(args.save_path,'start_epoch.txt'),'r') as f:
            start_epoch = int(f.readline())
        # read best_error
        with open(args.save_path/args.log_summary, 'r') as f:
            content = f.readlines()
        content = content[1:]
        if len(content) == 0:
            pass
        else:
            val_errors = []
            for line in content:
                val = line.split('\t')[-1][:-1]
                val_errors.append(val)
            val_errors = np.asarray(val_errors,dtype=np.float)
            best_error = np.min(val_errors)

    # import ipdb; ipdb.set_trace()
    cudnn.benchmark = True
    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)
    mask_net = torch.nn.DataParallel(mask_net)
    flow_net = torch.nn.DataParallel(flow_net)

    print('=> setting adam solver')

    parameters = chain(disp_net.parameters(), pose_net.parameters(), mask_net.parameters(), flow_net.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    if args.resume and (args.save_path/'optimizer_checkpoint.pth.tar').exists():
        print("=> loading optimizer from checkpoint")
        optimizer_weights = torch.load(args.save_path/'optimizer_checkpoint.pth.tar')
        optimizer.load_state_dict(optimizer_weights['state_dict'])

    if not args.resume:
        with open(args.save_path/args.log_summary, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss', 'validation_loss'])

        with open(args.save_path/args.log_full, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss', 'photo_cam_loss', 'photo_flow_loss', 'explainability_loss', 'smooth_loss'])

    if args.log_terminal:
        logger = TermLogger(n_epochs=start_epoch+args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
        logger.epoch_bar.start()
    else:
        logger=None

    for epoch in range(start_epoch,start_epoch+args.epochs):
        print("TRAINING EPOCH {}".format(epoch))

        if args.fix_flownet:
            for fparams in flow_net.parameters():
                fparams.requires_grad = False

        if args.fix_masknet:
            for fparams in mask_net.parameters():
                fparams.requires_grad = False

        if args.fix_posenet:
            for fparams in pose_net.parameters():
                fparams.requires_grad = False

        if args.fix_dispnet:
            for fparams in disp_net.parameters():
                fparams.requires_grad = False

        if not args.fix_dispnet and not args.fix_posenet:
            print("Training R = (DispNet, PoseNet)")
        if not args.fix_flownet:
            print("Training F = Flownet")
        if not args.fix_masknet:
            print("Training M = MaskNet")

        if args.log_terminal:
            logger.epoch_bar.update(epoch)
            logger.reset_train_bar()

        # train for one epoch
        train_loss = train(train_loader, disp_net, pose_net, mask_net, flow_net, optimizer, args.epoch_size, logger, training_writer)

        if args.log_terminal:
            logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))
            logger.reset_valid_bar()

        # evaluate on validation set
        if args.with_flow_gt:
            print('VALIDATING FLOW')
            flow_errors, flow_error_names = validate_flow_with_gt(val_flow_loader, disp_net, pose_net, mask_net, flow_net, epoch, logger, output_writers)
            
            error_string_flow = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(flow_error_names, flow_errors))

            if args.log_terminal:
                logger.valid_writer.write(' * Avg {}'.format(error_string_flow))

            for error, name in zip(flow_errors, flow_error_names): 
                training_writer.add_scalar(name, error, epoch)

        if args.with_depth_gt:
            print('VALIDATING DEPTH')
            depth_errors, depth_error_names = validate_depth_with_gt(val_loader, disp_net, epoch, logger, output_writers)

            error_string_depth = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(depth_error_names, depth_errors))

            if args.log_terminal:
                logger.valid_writer.write(' * Avg {}'.format(error_string_depth))

            for error, name in zip(depth_errors, depth_error_names):
                training_writer.add_scalar(name, error, epoch)

            # write a1 metric
            # training_writer.add_scalar(depth_error_names[3], depth_errors[3], epoch)

        if args.with_pose_gt:
            print('VALIDATING POSE')
            pose_errors, pose_error_names = validate_pose_with_gt(pose_net, framework)
            
            error_string_pose = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(pose_error_names, pose_errors))

            if args.log_terminal:
                logger.valid_writer.write(' * Avg {}'.format(error_string_pose))
            
            for error, name in zip(pose_errors, pose_error_names):
                training_writer.add_scalar(name, error, epoch)

        if args.with_mask_gt:
            print('VALIDATING MASK')
            mask_errors, mask_error_names = validate_mask_with_gt(disp_net, pose_net, mask_net, flow_net, val_mask_loader)

            error_string_mask = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(mask_error_names, mask_errors))

            if args.log_terminal:
                logger.valid_writer.write(' * Avg {}'.format(error_string_mask))
            else:
                print('Epoch {} completed'.format(epoch))

            for error, name in zip(mask_errors, mask_error_names):
                training_writer.add_scalar(name, error, epoch)


        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        if (args.fix_dispnet==False and args.fix_flownet==True and args.fix_posenet==False and args.fix_masknet==True): # training R
            decisive_error = float(depth_errors[3])*(-1)      # minus depth a1
        elif (args.fix_dispnet==True and args.fix_flownet==False and args.fix_posenet==True and args.fix_masknet==True): # training F
            decisive_error = float(flow_errors[-2])    # epe_non_rigid_with_gt_mask
        elif (args.fix_dispnet==True and args.fix_flownet==True and args.fix_posenet==True and args.fix_masknet==False): # training M
            decisive_error = float(flow_errors[3])     # percent outliers
        if best_error == -1000:
            best_error = decisive_error

        # if not args.fix_posenet: # R
        #     decisive_error = flow_errors[-2]    # epe_rigid_with_gt_mask
        # elif not args.fix_dispnet: # R
        #     decisive_error = depth_errors[0]      #depth abs_diff
        # elif not args.fix_flownet: # F
        #     decisive_error = flow_errors[-2]    #epe_non_rigid_with_gt_mask
        # elif not args.fix_masknet: # M
        #     decisive_error = flow_errors[3]     # percent outliers
        # if best_error < 0:
        #     best_error = decisive_error

        # error_names = ['epe_total', 'epe_rigid', 'epe_non_rigid', 'outliers', 
        # 'epe_total_with_gt_mask', 'epe_rigid_with_gt_mask', 
        # 'epe_non_rigid_with_gt_mask', 'outliers_gt_mask']

        # remember lowest error and save checkpoint
        is_best = decisive_error <= best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': mask_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': flow_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': optimizer.state_dict()
            },
            is_best)

        with open(os.path.join(args.save_path,'n_iter.txt'),'w') as f:
            f.write(str(n_iter))

        with open(os.path.join(args.save_path,'start_epoch.txt'),'w') as f:
            f.write(str(epoch+1))

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])

        # check if model should continue training or not
        if args.break_training_if_saturate:
            with open(args.save_path/args.log_summary, 'r') as f:
                content = f.readlines()
            if len(content) > 2:
                current_val_loss = float(content[-1].split('\t')[-1][:-1])
                previous_val_loss = float(content[-2].split('\t')[-1][:-1])
                change_rate = abs(current_val_loss-previous_val_loss)/(current_val_loss + epsilon)
                if change_rate < args.break_training_value_threshold:
                    saturation_count += 1
                else:
                    saturation_count = 0
            if saturation_count > args.break_training_count_threshold:
                print('Validation loss saturates for more than {} epochs => breaking training!!!'.format(args.break_training_count_threshold))
                break

    if args.log_terminal:
        logger.epoch_bar.finish()


def train(train_loader, disp_net, pose_net, mask_net, flow_net, optimizer, epoch_size, logger=None, train_writer=None):
    global args, n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1 = args.cam_photo_loss_weight
    w2 = args.mask_loss_weight
    w3 = args.smooth_loss_weight
    w4 = args.flow_photo_loss_weight
    w5 = args.consensus_loss_weight

    if args.robust:
        loss_camera = photometric_reconstruction_loss_robust
        loss_flow = photometric_flow_loss_robust
    else:
        loss_camera = photometric_reconstruction_loss
        loss_flow = photometric_flow_loss

    # switch to train mode
    disp_net.train()
    pose_net.train()
    mask_net.train()
    flow_net.train()

    end = time.time()

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = Variable(tgt_img.cuda())
        ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
        intrinsics_var = Variable(intrinsics.cuda())
        intrinsics_inv_var = Variable(intrinsics_inv.cuda())

        # compute output
        disparities = disp_net(tgt_img_var)
        if args.spatial_normalize:
            disparities = [spatial_normalize(disp) for disp in disparities]

        depth = [1/disp for disp in disparities]
        pose = pose_net(tgt_img_var, ref_imgs_var)
        explainability_mask = mask_net(tgt_img_var, ref_imgs_var)

        if args.flownet == 'Back2Future':
            flow_fwd, flow_bwd, _ = flow_net(tgt_img_var, ref_imgs_var[1:3])
        else:
            flow_fwd = flow_net(tgt_img_var, ref_imgs_var[2])
            flow_bwd = flow_net(tgt_img_var, ref_imgs_var[1])

        flow_cam = pose2flow(depth[0].squeeze(), pose[:,2], intrinsics_var, intrinsics_inv_var)     # pose[:,2] belongs to forward frame

        flows_cam_fwd = [pose2flow(depth_.squeeze(1), pose[:,2], intrinsics_var, intrinsics_inv_var) for depth_ in depth]
        flows_cam_bwd = [pose2flow(depth_.squeeze(1), pose[:,1], intrinsics_var, intrinsics_inv_var) for depth_ in depth]

        exp_masks_target = consensus_exp_masks(flows_cam_fwd, flows_cam_bwd, flow_fwd, flow_bwd, tgt_img_var, ref_imgs_var[2], ref_imgs_var[1], wssim=args.wssim, wrig=args.wrig, ws=args.smooth_loss_weight )

        rigidity_mask_fwd = [(flows_cam_fwd_ - flow_fwd_).abs() for flows_cam_fwd_, flow_fwd_ in zip(flows_cam_fwd, flow_fwd) ]#.normalize()
        rigidity_mask_bwd = [(flows_cam_bwd_ - flow_bwd_).abs() for flows_cam_bwd_, flow_bwd_ in zip(flows_cam_bwd, flow_bwd) ]#.normalize()

        if args.joint_mask_for_depth:
            explainability_mask_for_depth = compute_joint_mask_for_depth(explainability_mask, rigidity_mask_bwd, rigidity_mask_fwd)
        else:
            explainability_mask_for_depth = explainability_mask

        if args.no_non_rigid_mask:
            flow_exp_mask = [None for exp_mask in explainability_mask]
            if args.DEBUG:
                print('Using no masks for flow')
        else:
            flow_exp_mask = [1 - exp_mask[:,1:3] for exp_mask in explainability_mask]

        loss_1 = loss_camera(tgt_img_var, ref_imgs_var, intrinsics_var, intrinsics_inv_var,
                                        depth, explainability_mask_for_depth, pose, lambda_oob=args.lambda_oob, qch=args.qch, wssim=args.wssim)
        if w2 > 0:
            loss_2 = explainability_loss(explainability_mask) #+ 0.2*gaussian_explainability_loss(explainability_mask)
        else:
            loss_2 = 0

        if args.smoothness_type == "regular":
            loss_3 = smooth_loss(depth) + smooth_loss(flow_fwd) + smooth_loss(flow_bwd) + smooth_loss(explainability_mask)
        elif args.smoothness_type == "edgeaware":
            loss_3 = edge_aware_smoothness_loss(tgt_img_var, depth) + edge_aware_smoothness_loss(tgt_img_var, flow_fwd)
            loss_3 += edge_aware_smoothness_loss(tgt_img_var, flow_bwd) + edge_aware_smoothness_loss(tgt_img_var, explainability_mask)

        loss_4 = loss_flow(tgt_img_var, ref_imgs_var[1:3], [flow_bwd, flow_fwd], flow_exp_mask,
                                        lambda_oob=args.lambda_oob, qch=args.qch, wssim=args.wssim, use_occ_mask_at_scale=args.occ_mask_at_scale)

        loss_5 = consensus_depth_flow_mask(explainability_mask, rigidity_mask_bwd, rigidity_mask_fwd,
                                        exp_masks_target, exp_masks_target, THRESH=args.THRESH, wbce=args.wbce)

        # w1 = args.cam_photo_loss_weight
        # w2 = args.mask_loss_weight
        # w3 = args.smooth_loss_weight
        # w4 = args.flow_photo_loss_weight
        # w5 = args.consensus_loss_weight

        # loss_1 = loss_camera
        # loss_2 = explainability_loss
        # loss_3 = edge_aware_smoothness_loss
        # loss_4 = loss_flow
        # loss_5 = consensus_depth_flow_mask

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3 + w4*loss_4 + w5*loss_5

        if i > 0 and n_iter % args.print_freq == 0:
            train_writer.add_scalar('cam_photometric_error', loss_1.item(), n_iter)
            if w2 > 0:
                train_writer.add_scalar('explanability_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('flow_photometric_error', loss_4.item(), n_iter)
            train_writer.add_scalar('consensus_error', loss_5.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)


        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:

            train_writer.add_image('train Input', tensor2array(tgt_img[0]), n_iter)
            train_writer.add_image('train Cam Flow Output',
                                    flow_to_image(tensor2array(flow_cam.data[0].cpu())) , n_iter )

            for k,scaled_depth in enumerate(depth):
                train_writer.add_image('train Dispnet Output Normalized {}'.format(k),
                                       tensor2array(disparities[k].data[0].cpu(), max_value=None, colormap='bone'),
                                       n_iter)
                train_writer.add_image('train Depth Output {}'.format(k),
                                       tensor2array(1/disparities[k].data[0].cpu(), max_value=10),
                                       n_iter)
                train_writer.add_image('train Non Rigid Flow Output {}'.format(k),
                                        flow_to_image(tensor2array(flow_fwd[k].data[0].cpu())) , n_iter )
                train_writer.add_image('train Target Rigidity {}'.format(k),
                                        tensor2array((rigidity_mask_fwd[k]>args.THRESH).type_as(rigidity_mask_fwd[k]).data[0].cpu(), max_value=1, colormap='bone') , n_iter )

                b, _, h, w = scaled_depth.size()
                downscale = tgt_img_var.size(2)/h

                tgt_img_scaled = nn.functional.adaptive_avg_pool2d(tgt_img_var, (h, w))
                ref_imgs_scaled = [nn.functional.adaptive_avg_pool2d(ref_img, (h, w)) for ref_img in ref_imgs_var]

                intrinsics_scaled = torch.cat((intrinsics_var[:, 0:2]/downscale, intrinsics_var[:, 2:]), dim=1)
                intrinsics_scaled_inv = torch.cat((intrinsics_inv_var[:, :, 0:2]*downscale, intrinsics_inv_var[:, :, 2:]), dim=2)

                train_writer.add_image('train Non Rigid Warped Image {}'.format(k),
                                        tensor2array(flow_warp(ref_imgs_scaled[2],flow_fwd[k]).data[0].cpu()) , n_iter )

                # log warped images along with explainability mask
                for j,ref in enumerate(ref_imgs_scaled):
                    ref_warped = inverse_warp(ref, scaled_depth[:,0], pose[:,j],
                                              intrinsics_scaled, intrinsics_scaled_inv,
                                              rotation_mode=args.rotation_mode,
                                              padding_mode=args.padding_mode)[0]
                    train_writer.add_image('train Warped Outputs {} {}'.format(k,j), tensor2array(ref_warped.data.cpu()), n_iter)
                    train_writer.add_image('train Diff Outputs {} {}'.format(k,j), tensor2array(0.5*(tgt_img_scaled[0] - ref_warped).abs().data.cpu()), n_iter)
                    if explainability_mask[k] is not None:
                        train_writer.add_image('train Exp mask Outputs {} {}'.format(k,j), tensor2array(explainability_mask[k][0,j].data.cpu(), max_value=1, colormap='bone'), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item() if w2 > 0 else 0, loss_3.item(), loss_4.item()])
        if args.log_terminal:
            logger.train_bar.update(i+1)
            if i % args.print_freq == 0:
                logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]

def validate_depth_with_gt(val_loader, disp_net, epoch, logger, output_writers=[]):
    global args
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()

    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img_var = Variable(tgt_img.cuda(), volatile=True)
        output_disp = disp_net(tgt_img_var)
        if args.spatial_normalize:
            output_disp = spatial_normalize(output_disp)

        output_depth = 1/output_disp

        depth = depth.cuda()

        # compute output

        if log_outputs and i % 100 == 0 and i/100 < len(output_writers):
            index = int(i//100)
            if epoch == start_epoch:
                output_writers[index].add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0].cpu()
                output_writers[index].add_image('val target Depth', tensor2array(depth_to_show, max_value=10), epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0,10)
                output_writers[index].add_image('val target Disparity Normalized', tensor2array(disp_to_show, max_value=None, colormap='bone'), epoch)

            output_writers[index].add_image('val Dispnet Output Normalized', tensor2array(output_disp.data[0].cpu(), max_value=None, colormap='bone'), epoch)
            output_writers[index].add_image('val Depth Output', tensor2array(output_depth.data[0].cpu(), max_value=10), epoch)

        errors.update(compute_errors(depth, output_depth.data.squeeze(1)))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if args.log_terminal:
            logger.valid_bar.update(i)
            if i % args.print_freq == 0:
                logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    if args.log_terminal:
        logger.valid_bar.update(len(val_loader))

    # # return only a1
    # return_errors = [errors.avg[3]]
    # return_errors_names = [error_names[3]]

    return errors.avg, error_names

def validate_flow_with_gt(val_loader, disp_net, pose_net, mask_net, flow_net, epoch, logger, output_writers=[]):
    global args
    batch_time = AverageMeter()
    error_names = ['epe_total', 'epe_rigid', 'epe_non_rigid', 'outliers', 'epe_total_with_gt_mask', 'epe_rigid_with_gt_mask', 'epe_non_rigid_with_gt_mask', 'outliers_gt_mask']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    flow_net.eval()

    end = time.time()

    poses = np.zeros(((len(val_loader)-1) * 1 * (args.sequence_length-1),6))

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt) in enumerate(val_loader):
        with torch.no_grad():
            tgt_img_var = Variable(tgt_img.cuda())
            ref_imgs_var = [Variable(img.cuda()) for img in ref_imgs]
            intrinsics_var = Variable(intrinsics.cuda())
            intrinsics_inv_var = Variable(intrinsics_inv.cuda())

            flow_gt_var = Variable(flow_gt.cuda())
            obj_map_gt_var = Variable(obj_map_gt.cuda())

        # compute output
        disp = disp_net(tgt_img_var)
        if args.spatial_normalize:
            disp = spatial_normalize(disp)

        depth = 1/disp
        pose = pose_net(tgt_img_var, ref_imgs_var)
        explainability_mask = mask_net(tgt_img_var, ref_imgs_var)

        if args.flownet == 'Back2Future':
            flow_fwd, flow_bwd, _ = flow_net(tgt_img_var, ref_imgs_var[1:3])
        else:
            flow_fwd = flow_net(tgt_img_var, ref_imgs_var[2])
            flow_bwd = flow_net(tgt_img_var, ref_imgs_var[1])

        if args.DEBUG:
            flow_fwd_x = flow_fwd[:,0].view(-1).abs().data
            print("Flow Fwd Median: ", flow_fwd_x.median())
            flow_gt_var_x = flow_gt_var[:,0].view(-1).abs().data
            print("Flow GT Median: ", flow_gt_var_x.index_select(0, flow_gt_var_x.nonzero().view(-1)).median())
        flow_cam = pose2flow(depth.squeeze(1), pose[:,2], intrinsics_var, intrinsics_inv_var)
        oob_rigid = flow2oob(flow_cam)
        oob_non_rigid = flow2oob(flow_fwd)

        rigidity_mask = 1 - (1-explainability_mask[:,1])*(1-explainability_mask[:,2]).unsqueeze(1) > 0.5

        rigidity_mask_census_soft = (flow_cam - flow_fwd).abs()#.normalize()
        rigidity_mask_census_u = rigidity_mask_census_soft[:,0] < args.THRESH
        rigidity_mask_census_v = rigidity_mask_census_soft[:,1] < args.THRESH
        rigidity_mask_census = (rigidity_mask_census_u).type_as(flow_fwd) * (rigidity_mask_census_v).type_as(flow_fwd)

        rigidity_mask_combined = 1 - (1-rigidity_mask.type_as(explainability_mask))*(1-rigidity_mask_census.type_as(explainability_mask))

        flow_fwd_non_rigid = (rigidity_mask_combined<=args.THRESH).type_as(flow_fwd).expand_as(flow_fwd) * flow_fwd
        flow_fwd_rigid = (rigidity_mask_combined>args.THRESH).type_as(flow_fwd).expand_as(flow_fwd) * flow_cam
        total_flow = flow_fwd_rigid + flow_fwd_non_rigid

        obj_map_gt_var_expanded = obj_map_gt_var.unsqueeze(1).type_as(flow_fwd)

        if log_outputs and i % 10 == 0 and i/10 < len(output_writers):
            index = int(i//10)
            if epoch == start_epoch:
                output_writers[index].add_image('val flow Input', tensor2array(tgt_img[0]), 0)
                flow_to_show = flow_gt[0][:2,:,:].cpu()
                output_writers[index].add_image('val target Flow', flow_to_image(tensor2array(flow_to_show)), epoch)

            output_writers[index].add_image('val Total Flow Output', flow_to_image(tensor2array(total_flow.data[0].cpu())), epoch)
            output_writers[index].add_image('val Rigid Flow Output', flow_to_image(tensor2array(flow_fwd_rigid.data[0].cpu())), epoch)
            output_writers[index].add_image('val Non-rigid Flow Output', flow_to_image(tensor2array(flow_fwd_non_rigid.data[0].cpu())), epoch)
            output_writers[index].add_image('val Out of Bound (Rigid)', tensor2array(oob_rigid.type(torch.FloatTensor).data[0].cpu(), max_value=1, colormap='bone'), epoch)
            output_writers[index].add_scalar('val Mean oob (Rigid)', oob_rigid.type(torch.FloatTensor).sum(), epoch)
            output_writers[index].add_image('val Out of Bound (Non-Rigid)', tensor2array(oob_non_rigid.type(torch.FloatTensor).data[0].cpu(), max_value=1, colormap='bone'), epoch)
            output_writers[index].add_scalar('val Mean oob (Non-Rigid)', oob_non_rigid.type(torch.FloatTensor).sum(), epoch)
            output_writers[index].add_image('val Cam Flow Errors', tensor2array(flow_diff(flow_gt_var, flow_cam).data[0].cpu()), epoch)
            output_writers[index].add_image('val Rigidity Mask', tensor2array(rigidity_mask.data[0].cpu(), max_value=1, colormap='bone'), epoch)
            output_writers[index].add_image('val Rigidity Mask Census', tensor2array(rigidity_mask_census.data[0].cpu(), max_value=1, colormap='bone'), epoch)

            for j,ref in enumerate(ref_imgs_var):
                ref_warped = inverse_warp(ref[:1], depth[:1,0], pose[:1,j],
                                          intrinsics_var[:1], intrinsics_inv_var[:1],
                                          rotation_mode=args.rotation_mode,
                                          padding_mode=args.padding_mode)[0]

                output_writers[index].add_image('val Warped Outputs {}'.format(j), tensor2array(ref_warped.data.cpu()), epoch)
                output_writers[index].add_image('val Diff Outputs {}'.format(j), tensor2array(0.5*(tgt_img_var[0] - ref_warped).abs().data.cpu()), epoch)
                if explainability_mask is not None:
                    output_writers[index].add_image('val Exp mask Outputs {}'.format(j), tensor2array(explainability_mask[0,j].data.cpu(), max_value=1, colormap='bone'), epoch)

            if args.DEBUG:
                # Check if pose2flow is consistant with inverse warp
                ref_warped_from_depth = inverse_warp(ref_imgs_var[2][:1], depth[:1,0], pose[:1,2],
                            intrinsics_var[:1], intrinsics_inv_var[:1], rotation_mode=args.rotation_mode,
                            padding_mode=args.padding_mode)[0]
                ref_warped_from_cam_flow = flow_warp(ref_imgs_var[2][:1], flow_cam)[0]
                print("DEBUG_INFO: Inverse_warp vs pose2flow",torch.mean(torch.abs(ref_warped_from_depth-ref_warped_from_cam_flow)).item())
                output_writers[index].add_image('val Warped Outputs from Cam Flow', tensor2array(ref_warped_from_cam_flow.data.cpu()), epoch)
                output_writers[index].add_image('val Warped Outputs from inverse warp', tensor2array(ref_warped_from_depth.data.cpu()), epoch)

        if log_outputs and i < len(val_loader)-1:
            step = args.sequence_length-1
            poses[i * step:(i+1) * step] = pose.data.cpu().view(-1,6).numpy()


        if np.isnan(flow_gt.sum().item()) or np.isnan(total_flow.data.sum().item()):
            print('NaN encountered')
        _epe_errors = compute_all_epes(flow_gt_var, flow_cam, flow_fwd, rigidity_mask_combined) + compute_all_epes(flow_gt_var, flow_cam, flow_fwd, (1-obj_map_gt_var_expanded))
        errors.update(_epe_errors)

        if args.DEBUG:
            print("DEBUG_INFO: EPE errors: ", _epe_errors )
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if log_outputs:
        output_writers[0].add_histogram('val poses_tx', poses[:,0], epoch)
        output_writers[0].add_histogram('val poses_ty', poses[:,1], epoch)
        output_writers[0].add_histogram('val poses_tz', poses[:,2], epoch)
        if args.rotation_mode == 'euler':
            rot_coeffs = ['rx', 'ry', 'rz']
        elif args.rotation_mode == 'quat':
            rot_coeffs = ['qx', 'qy', 'qz']
        output_writers[0].add_histogram('val poses_{}'.format(rot_coeffs[0]), poses[:,3], epoch)
        output_writers[0].add_histogram('val poses_{}'.format(rot_coeffs[1]), poses[:,4], epoch)
        output_writers[0].add_histogram('val poses_{}'.format(rot_coeffs[2]), poses[:,5], epoch)

    if args.DEBUG:
        print("DEBUG_INFO =================>")
        print("DEBUG_INFO: Average EPE : ", errors.avg )
        print("DEBUG_INFO =================>")
        print("DEBUG_INFO =================>")
        print("DEBUG_INFO =================>")

    return errors.avg, error_names

def validate_pose_with_gt(pose_net, framework):
    def compute_pose_error(gt, pred):
        RE = 0
        snippet_length = gt.shape[0]
        scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
        ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
        for gt_pose, pred_pose in zip(gt, pred):
            # Residual matrix to which we compute angle's sin and cos
            R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
            s = np.linalg.norm([R[0,1]-R[1,0],
                                R[1,2]-R[2,1],
                                R[0,2]-R[2,0]])
            c = np.trace(R) - 1
            # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
            RE += np.arctan2(s,c)

        return ATE/snippet_length, RE/snippet_length
    from scipy.misc import imresize
    no_resize = False
    img_width = 832
    img_height = 256
    min_depth=1e-3
    max_depth=80
    img_exts=['png', 'jpg', 'bmp']
    rotation_mode='euler'

    errors = np.zeros((len(framework), 2), np.float32)

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not no_resize) and (h != img_height or w != img_width):
            imgs = [imresize(img, (img_height, img_width)).astype(np.float32) for img in imgs]

        imgs = [np.transpose(img, (2,0,1)) for img in imgs]

        ref_imgs_var = []
        for i, img in enumerate(imgs):
            img = torch.from_numpy(img).unsqueeze(0)
            img = ((img/255 - 0.5)/0.5).cuda()
            img_var = Variable(img, volatile=True)
            if i == len(imgs)//2:
                tgt_img_var = img_var
            else:
                ref_imgs_var.append(Variable(img, volatile=True))

        poses = pose_net(tgt_img_var, ref_imgs_var)

        poses = poses.cpu().data[0]
        poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])

        inv_transform_matrices = pose_vec2mat(Variable(poses), rotation_mode=rotation_mode).data.numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']

    return mean_errors, error_names

def validate_mask_with_gt(disp_net,pose_net,mask_net,flow_net,val_mask_loader):

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

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()
    mask_net.eval()
    flow_net.eval()

    error_names = ['tp_0', 'fp_0', 'fn_0', 'tp_1', 'fp_1', 'fn_1']
    errors = AverageMeter(i=len(error_names))
    errors_census = AverageMeter(i=len(error_names))
    errors_bare = AverageMeter(i=len(error_names))

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, flow_gt, obj_map_gt, semantic_map_gt) in enumerate(tqdm(val_mask_loader)):
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
    print("Results Full Model")
    print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    print("Errors \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou, bg_iou, fg_iou))

    print("Results Census only")
    print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    print("Errors \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou_census, bg_iou_census, fg_iou_census))

    print("Results Bare")
    print("\t {:>10}, {:>10}, {:>10} ".format('iou', 'bg_iou', 'fg_iou'))
    print("Errors \t {:10.4f}, {:10.4f} {:10.4f}".format(avg_iou_bare, bg_iou_bare, fg_iou_bare))
    return errors_return, error_names



if __name__ == '__main__':
    import sys
    with open("/content/drive/MyDrive/VinAI/Motion segmentation/experiment_recorder.md", "a") as f:
        f.write('\n python3 ' + ' '.join(sys.argv))
    main()
