# Reference : https://github.com/karfly/learnable-triangulation-pytorch

import numpy as np

import torch
import torch.nn as nn

def MSEsoft(pred, gt, threshold=20**2):
    diff = nn.MSELoss(reduction='none')(pred, gt)
    diff[diff > threshold] = (diff[diff > threshold]**0.1) * (threshold**0.9)
    return torch.mean(diff)

def MAELoss(pred, gt):
    loss = nn.L1Loss()
    return loss(pred, gt)

def MPJPE(keypoints_3d_pred, keypoints_3d_gt): ################### TODO
    '''Mean Per Joint Position Error'''
    keypoints_3d_pred = np.concatenate(keypoints_3d_pred, axis=0)
    keypoints_3d_gt   = np.concatenate(keypoints_3d_gt,   axis=0)

    # L2 distance
    per_pose_error = np.sqrt(((keypoints_3d_gt - keypoints_3d_pred)**2).sum(2)).mean(1)

    # calculate per action
    ################### TODO
