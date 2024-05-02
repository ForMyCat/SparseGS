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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import matplotlib.pyplot as plt
import os

def build_rot_y(th):
        return torch.Tensor([
        [np.cos(th), 0, -np.sin(th)],
        [0, 1, 0],
        [np.sin(th), 0, np.cos(th)]])

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, depth, gt_alpha_mask,
                 image_name, uid, warp_mask, K, src_R, src_T, src_uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.warp_mask = warp_mask
        self.K = K
        self.src_R = src_R
        self.src_T = src_T
        self.src_uid = src_uid

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.depth = depth
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        def gen_hemisphere_view(self, theta, center):
            cam_focus = torch.Tensor(self.T) - center
            rot_y = build_rot_y(theta)
            t_n = torch.Tensor(rot_y @ cam_focus + center)

            E_ref = torch.Tensor(getWorld2View2(self.R, self.T))
            E_n = torch.Tensor(getWorld2View2(self.R, t_n))
            ref_img = torch.Tensor(self.original_image).cpu()
            ref_depth = torch.Tensor(self.depth)
            K_ref = torch.Tensor(self.cam_intr)

            new_look_at = center - t_n
            new_look_at = new_look_at / torch.linalg.norm(new_look_at)
            new_right = rot_y @ self.R[:,1]
            new_right = new_right / torch.linalg.norm(new_right)
            new_up = torch.cross(new_look_at.float(), new_right.float())

            R_n = torch.stack((new_right, new_up, new_look_at), dim=1)

    def generate_warp_gt(self, rendered_depth_min, rendered_depth_max):
        def depth_warping(img, depth_map, K, R_A, T_A, R_B, T_B,  rendered_depth_min, rendered_depth_max):
            ''' 
            img: src image
            K: Intrinsics
            R_A, T_A: Extrinsics matrix of the src image
            R_B, T_B: Extrinsics matrix of the target image
            depth_map: Mono Depth from the src camera
            rendered_depth_min/_max: 3dgs rendered depth min/max from the src camera in COLMAP coordinate
            '''
            # Calculate the transformation from A to B
            R_A = R_A.T
            R_B = R_B.T
            img = img.detach().cpu().numpy().transpose(1,2,0)
            depth_map = depth_map.detach().cpu().numpy()
            # Scale monodepth to COLMAP coordinate
            scaled_depth = (depth_map - depth_map.min())/(depth_map.max() - depth_map.min())
            scaled_depth = scaled_depth * (rendered_depth_max - rendered_depth_min) + rendered_depth_min
            R_AB = R_B @ np.linalg.inv(R_A)
            T_AB = T_B - (R_AB @ T_A)
            K = K.clone().cpu().numpy()
            K_inv = np.linalg.inv(K)
            height, width = img.shape[:2]
            warp_mask = np.zeros((height, width), dtype=np.double)
            warped_img = np.zeros_like(img)

            for y in range(height):
                for x in range(width):
                    Z = scaled_depth[y, x]
                    
                    xy_homog = np.array([x, y, 1])

                    # Convert to COLMAP coordinate system
                    xy_normalized = K_inv @ xy_homog
                    # Backproject to 3D 
                    P_A = Z * xy_normalized
                    # A-B Transformation matrix
                    P_B = R_AB @ P_A[:3] + T_AB
                    # Project onto camera B's image plane
                    xy_b_homog = K @ P_B
                    # Normalize to get the pixel coordinates
                    x_b, y_b = (xy_b_homog / xy_b_homog[2])[:2]
                    x_b, y_b = int(round(x_b)), int(round(y_b))
                    # Check bounds
                    if 0 <= x_b < width and 0 <= y_b < height:
                        warped_img[y_b, x_b] = img[y, x]
                        warp_mask[y_b, x_b] = 1.0
            out_img = warped_img
            return out_img.transpose(2,0,1), warp_mask

        src_img = self.original_image # Torch tensor
        scaled_depth = self.depth # Torch tensor
        src_R = self.src_R
        src_T = self.src_T
        trg_R = self.R
        trg_T = self.T
        K = self.K
        uid = self.uid
        print('Warping',self.src_uid,'to',self.uid,'dp min:',round(rendered_depth_min,5),'dp max:',round(rendered_depth_max,5))
        warped_img, warped_mask = depth_warping(src_img, scaled_depth, K, src_R, src_T, trg_R, trg_T, rendered_depth_min, rendered_depth_max)

        warped_mask = warped_mask.astype(np.double)
        warped_img = torch.from_numpy(warped_img).to(torch.float32).to('cuda').detach()
        warped_mask = torch.from_numpy(warped_mask).to(torch.float32).to('cuda').detach()
        self.original_image = warped_img
        self.warp_mask = warped_mask




class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

