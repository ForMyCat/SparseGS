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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, NP_resize
from utils.graphics_utils import fov2focal
import torch

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))


    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    # ================================= Resize depth map ==========================================================
    resized_depth = None
    if cam_info.depth is not None:
        resized_depth = NP_resize(cam_info.depth, resolution)
        resized_depth = torch.Tensor((resized_depth - resized_depth.min())/(resized_depth.max() - resized_depth.min())).cuda()
    # =============================================================================================================

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, warp_mask = None,
                  image=gt_image, depth=resized_depth, gt_alpha_mask=loaded_mask, K=None, src_R=None, src_T=None,src_uid = id,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

# def depth_warping(img, depth_map, K, R_A, T_A, R_B, T_B,  rendered_depth_min, rendered_depth_max):
#     ''' 
#     img: src image
#     K: Intrinsics
#     R_A, T_A: Extrinsics matrix of the src image
#     R_B, T_B: Extrinsics matrix of the target image
#     depth_map: Mono Depth from the src camera
#     rendered_depth_min/_max: 3dgs rendered depth min/max from the src camera in COLMAP coordinate
#     '''
#     # Calculate the transformation from A to B
#     R_A = R_A.T
#     R_B = R_B.T
#     img = img.detach().cpu().numpy().transpose(1,2,0)
#     # img = (img*255).astype(np.double)
#     depth_map = depth_map.detach().cpu().numpy()
#     rendered_depth = rendered_depth
#     # rendered_depth = np.load('render_views/vid_images/bonsai_12_warp_half_more_angle_higher_pearson/iteration_7000/depth_value/'+str(cam_uid)+'.npy')
#     rendered_depth_np = rendered_depth.squeeze(0)

#     scaled_depth = (depth_map - depth_map.min())/(depth_map.max() - depth_map.min())
#     scaled_depth = scaled_depth * (rendered_depth_max - rendered_depth_min) + rendered_depth_min

#     # K = K.clone().cpu().detach().numpy()
#     # K = np.array([
#     # [3222.7010797592447/2, 0, (3118-1) / 4],
#     # [0, 3222.7010797592447/2, (2078-1) / 4],
#     # [0, 0, 1.0]
#     # ])

#     R_AB = R_B @ np.linalg.inv(R_A)
#     T_AB = T_B - (R_AB @ T_A)
    
#     K_inv = np.linalg.inv(K)
#     height, width = img.shape[:2]
#     warp_mask = np.zeros((height, width), dtype=np.double)
#     warped_img = np.zeros_like(img)
#     for y in range(height):
#         for x in range(width):
#             Z = scaled_depth[y, x]
#             xy_homog = np.array([x, y, 1])

#             # Convert to COLMAP coordinate system
#             xy_normalized = K_inv @ xy_homog

#             # Backproject to 3D 
#             P_A = Z * xy_normalized

#             # A-B Transformation matrix
#             P_B = R_AB @ P_A[:3] + T_AB

#             # Project onto camera B's image plane
#             xy_b_homog = K @ P_B

#             # Normalize to get the pixel coordinates
#             x_b, y_b = (xy_b_homog / xy_b_homog[2])[:2]
#             x_b, y_b = int(round(x_b)), int(round(y_b))
            
#             # Check bounds
#             if 0 <= x_b < width and 0 <= y_b < height:
#                 warped_img[y_b, x_b] = img[y, x]
#                 warp_mask[y_b, x_b] = 1.0


#     out_img = warped_img

#     return out_img.transpose(2,0,1), warp_mask

