import torch
from torch.autograd import Variable

from scipy.misc import imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

import models
from inverse_warp import pose_vec2mat


parser = argparse.ArgumentParser(description='Script for PoseNet testing with corresponding groundTruth from KITTI Odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("pretrained_posenet", type=str, help="pretrained PoseNet path")
parser.add_argument("--posenet", type=str, default="PoseNetB6", help="PoseNet model path")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)

parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory") # '/content/drive/MyDrive/Dự án/KITTI Dataset/Odometry/dataset'
parser.add_argument("--sequences", default=['09'], type=str, nargs='*', help="sequences to test")
parser.add_argument("--output-dir", default=None, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)


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

    print('{} snippets to test'.format(len(framework)))
    errors = np.zeros((len(framework), 2), np.float32)

    for j, sample in enumerate(tqdm(framework)):
        imgs = sample['imgs']

        h,w,_ = imgs[0].shape
        if (not args.no_resize) and (h != args.img_height or w != args.img_width):
            imgs = [imresize(img, (args.img_height, args.img_width)).astype(np.float32) for img in imgs]

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

        if args.posenet in ["PoseNet6", "PoseNetB6"]:
            poses = pose_net(tgt_img_var, ref_imgs_var)
        else:
            _, poses = pose_net(tgt_img_var, ref_imgs_var)

        poses = poses.cpu().data[0]
        poses = torch.cat([poses[:len(imgs)//2], torch.zeros(1,6).float(), poses[len(imgs)//2:]])

        inv_transform_matrices = pose_vec2mat(Variable(poses), rotation_mode=args.rotation_mode).data.numpy().astype(np.float64)

        rot_matrices = np.linalg.inv(inv_transform_matrices[:,:,:3])
        tr_vectors = -rot_matrices @ inv_transform_matrices[:,:,-1:]

        transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

        first_inv_transform = inv_transform_matrices[0]
        final_poses = first_inv_transform[:,:3] @ transform_matrices
        final_poses[:,:,-1:] += first_inv_transform[:,-1:]

        if args.output_dir is not None:
            predictions_array[j] = final_poses

        ATE, RE = compute_pose_error(sample['poses'], final_poses)
        errors[j] = ATE, RE

    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']

    return error_names, mean_errors
