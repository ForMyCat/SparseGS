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
import numpy as np
from scene.generate_new_cam import rotate_camera_poses, create_cam_obj, calculate_average_up_vector
import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import setup_ellipse_sampling, viewmatrix


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], step=1, max_cameras=None, mode='train'):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.ft_cameras = {}
        self.ellipse_params = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, step=step, max_cameras=max_cameras, load_depth=(not args.no_load_depth))
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]


        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            transform, center, up, low, high, z_low, z_high, ts, t_thetas = setup_ellipse_sampling(self.train_cameras[resolution_scale])       
            self.ellipse_params[resolution_scale] = {"transform": transform, "center": center, "up": up, "low": low, "high": high, "z_low": z_low, "z_high": z_high, "ts": ts, "t_thetas": t_thetas}
            if mode != 'eval':
            # # Rotate around average up vector
                train_cam_list = self.train_cameras[resolution_scale]
                num_poses = len(train_cam_list)
                R_list = list()
                T_list = list()
                for cam in train_cam_list:
                    R_list.append(cam.R)
                    T_list.append(cam.T)
                R_matrices = np.stack(R_list)
                T_vectors = np.stack(T_list)
                avg_up_vector = calculate_average_up_vector(R_matrices)
                
                cam_focal_dict = dict()
                for i in json_cams:
                    cam_focal_dict[i['id']] = (i['fx'],i['fy'])

                # Warping camera angles
                novel_cam_append_list = list()
                for d in [-2.5, 2.5]:
                    degree = d
                    new_R_matrices, new_T_vectors = rotate_camera_poses(avg_up_vector, R_matrices,
                                                                        T_vectors, degree * np.pi / 180)
                    for idx in range(num_poses):
                        train_cam = train_cam_list[idx]
                        novel_cam_append_list.append(create_cam_obj(train_cam,degree,
                                                                    new_R_matrices[idx,:,:],new_T_vectors[idx,:],args.resolution,cam_focal_dict))
                self.ft_cameras[resolution_scale] = novel_cam_append_list



            

        
        cam_centers = [x.camera_center for x in self.train_cameras[1.0]]
        cam_centers = torch.stack(cam_centers)
        gaussians.avg_cam_center = torch.mean(cam_centers, dim=0)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getFtCameras(self, scale=1.0):
        return self.ft_cameras[scale]
    
    def getRandEllipsePose(self, pose_idx, std, scale=1.0, z_variation=0.0):
        params = self.ellipse_params[scale]
        transform, center, up, low, high, z_low, z_high, ts, t_thetas = params["transform"], params["center"], params["up"], params["low"], params["high"], params["z_low"], params["z_high"], params["ts"], params["t_thetas"]
        
        theta_view = t_thetas[pose_idx]
        
        if std <= 0:    
            z_rand = np.random.uniform(z_low[2], z_high[2])
            theta = np.random.uniform(0, 2*np.pi)
        else:
            z_range = z_high[2] - z_low[2]
            z_rand =  ts[pose_idx][2] + float(np.random.normal(0,z_variation*z_range,1)[0])
            theta = theta_view + float(np.random.normal(0,std,1)[0])
        x =  (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5))
        y = (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5))
        position = np.stack([
           x,
            y,
            z_rand,
        ], -1)

        #position = np.matmul(rot_mat, ts[pose_idx])
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(position - center, up, position)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_pose = np.linalg.inv(render_pose)
        return render_pose