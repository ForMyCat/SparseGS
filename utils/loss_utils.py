#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from guidance.sd_utils import StableDiffusion
import math
from torchmetrics import PearsonCorrCoef
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn


pearson = PearsonCorrCoef().cuda()

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def mask_l1_loss(network_output, gt, mask):
    num_unmasked_elements = torch.sum(mask)
    sum_masked_diff = torch.abs((network_output - gt) * mask).sum()
    return sum_masked_diff/num_unmasked_elements

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def pearson_depth_loss(depth_src, depth_target):
    #co = pearson(depth_src.reshape(-1), depth_target.reshape(-1))

    src = depth_src - depth_src.mean()
    target = depth_target - depth_target.mean()

    src = src / (src.std() + 1e-6)
    target = target / (target.std() + 1e-6)

    co = (src * target).mean()
    assert not torch.any(torch.isnan(co))
    return 1 - co


def local_pearson_loss(depth_src, depth_target, box_p, p_corr):
        # Randomly select patch, top left corner of the patch (x_0,y_0) has to be 0 <= x_0 <= max_h, 0 <= y_0 <= max_w
        num_box_h = math.floor(depth_src.shape[0]/box_p)
        num_box_w = math.floor(depth_src.shape[1]/box_p)
        max_h = depth_src.shape[0] - box_p
        max_w = depth_src.shape[1] - box_p
        _loss = torch.tensor(0.0,device='cuda')
        n_corr = int(p_corr * num_box_h * num_box_w)
        x_0 = torch.randint(0, max_h, size=(n_corr,), device = 'cuda')
        y_0 = torch.randint(0, max_w, size=(n_corr,), device = 'cuda')
        x_1 = x_0 + box_p
        y_1 = y_0 + box_p
        _loss = torch.tensor(0.0,device='cuda')
        for i in range(len(x_0)):
            _loss += pearson_depth_loss(depth_src[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1), depth_target[x_0[i]:x_1[i],y_0[i]:y_1[i]].reshape(-1))
        return _loss/n_corr
