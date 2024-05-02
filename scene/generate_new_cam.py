import numpy as np
import torch
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, fov2focal
import matplotlib.pyplot as plt

def build_rot_y(th):
    return np.array([
    [np.cos(th), 0, -np.sin(th)],
    [0, 1, 0],
    [np.sin(th), 0, np.cos(th)]])

# def gen_new_cam_sai(cam, degree, center):
#     uid = cam.uid + 0.01*degree
#     theta = degree *np.pi/180 
#     cam_focus = np.array(torch.tensor(cam.T) - center)
#     rot_y = build_rot_y(theta)
#     #R_n = rot_y.numpy @ cam.R #they store the transpose
#     t_n = rot_y @ cam_focus + np.array(center)

#     E_ref = torch.Tensor(getWorld2View2(cam.R, cam.T))
    
#     # ref_img = torch.Tensor(cam.original_image).cpu()
#     # ref_depth = torch.Tensor(cam.depth)
#     # K_ref = torch.Tensor(cam.cam_intr)

#     new_look_at = np.array(center) - t_n
#     new_look_at = new_look_at / np.linalg.norm(new_look_at)
#     new_right = rot_y @ cam.R[:,0]
#     new_right = new_right / np.linalg.norm(new_right)
#     new_up = np.cross(new_look_at, new_right)

#     R_n = np.stack((new_right, new_up, new_look_at), axis=1)
#     R_n = cam.R @ rot_y 
#     E_n = torch.Tensor(getWorld2View2(R_n, t_n))

#     return Camera(colmap_id= None, R=R_n, T=t_n , FoVx=cam.FoVx, FoVy=cam.FoVy,  
#         image=cam.original_image,  vit_cam = True, vit_feature = cam.vit_feature,
#         image_name=None, uid=uid, gt_alpha_mask = None,
#             data_device = "cuda"
#         )

def gen_new_cam(cam, degree, rot_axis = 'y'):
    uid = cam.uid + 0.01*degree
    theta = degree *np.pi/180 

    old_R = cam.R.astype(np.float32)
    old_T = cam.T.astype(np.float32)


    #Generate new camera params
    image_name = cam.image_name + '_' + str(0.01*degree) + rot_axis
    if rot_axis == 'y':
        uid = cam.uid + 0.01*degree
    elif rot_axis == 'x':
        uid = cam.uid + 0.001*degree
    else:
        uid = cam.uid + 0.0001*degree
    colmap_id = cam.colmap_id + 0.01*degree


    R_n, t_n = gen_rotation_extr(old_R, old_T, degree = degree, rot_axis = rot_axis)

    return Camera(colmap_id= None, R=R_n, T=t_n , FoVx=cam.FoVx, FoVy=cam.FoVy,  
        image=cam.original_image,  ft_cam = True, vit_feature = cam.vit_feature,
        image_name=None, uid=uid, gt_alpha_mask = None,
            data_device = "cuda"
        )

def gen_rotation_extr(R, T, degree = 1, rot_axis = 'y'):
    '''
    This function takes in camera extrinsics and an angle
    Return the new extrinsics 
    Not Spherical Warp
    '''
    th = degree*np.pi/180 
    if rot_axis == 'y':
        rot_mat =  [
        [np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)]]
    elif rot_axis == 'x':
        rot_mat = [
        [1, 0, 0],
        [0, np.cos(th), -np.sin(th)],
        [0, np.sin(th), np.cos(th)]]
    elif rot_axis == 'z':
        rot_mat = [
        [np.cos(th), -np.sin(th), 0],
        [np.sin(th), np.cos(th), 0],
        [0, 1, 0]]
    
    new_R = np.matmul(rot_mat ,R)
    new_T = np.matmul(rot_mat ,T)
    return new_R, new_T

def gen_translation_extr(T, distance = 1.0, T_axis = 'y'):
    '''
    This function takes in camera extrinsics and a distance
    Return the new extrinsics 
    Translation Only
    '''
    if T_axis == 'y':
        new_T = T + np.array([0.0 , distance, 0.0])
    elif T_axis == 'x':
        new_T = T + np.array([distance , 0.0, 0.0])
    elif T_axis == 'z':
        new_T = T + np.array([0.0 , 0.0, distance])

    return new_T

def calculate_average_up_vector(rotation_matrices):
    up_vectors = rotation_matrices[:, :, 1]  # Assuming the up vector is the second column
    average_up_vector = np.mean(up_vectors, axis=0)
    normalized_average_up_vector = average_up_vector / np.linalg.norm(average_up_vector)
    return normalized_average_up_vector

def construct_rotation_matrix(axis, theta):
    axis /= np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0) #  Rodrigues' rotation formula
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def rotate_camera_poses(axis, rotation_matrices, translation_vectors, theta):
    rotation_matrix_avg = construct_rotation_matrix(axis, theta)  # Rotate around the avg y-axis
    new_rotation_matrices = np.matmul(rotation_matrix_avg, rotation_matrices)
    new_translation_vectors = translation_vectors
    return new_rotation_matrices, new_translation_vectors

def create_cam_obj(cam, degree, R, T, scaling_factor,cam_focal_dict):
    src_img = cam.original_image.clone().detach() # Torch tensor
    scaled_depth = cam.depth.clone().detach() # Torch tensor
    src_R = cam.R
    src_T = cam.T
    K = torch.from_numpy(np.array([
    [cam_focal_dict[cam.uid][0]/scaling_factor, 0, (src_img.shape[2]-1) / 2],
    [0, cam_focal_dict[cam.uid][1]/scaling_factor, (src_img.shape[1]-1) / 2],
    [0, 0, 1.0]
    ])).to('cuda')
    uid = cam.uid + 0.01*degree

    return Camera(colmap_id= None, R=R, T=T, FoVx = cam.FoVx, FoVy = cam.FoVy,  
        image=src_img, depth=scaled_depth, K = K, src_R = src_R, src_T = src_T, src_uid = cam.uid,
        image_name=None, uid=uid, gt_alpha_mask = None, warp_mask = None,
            data_device = "cuda"
        )




def calculate_average_camera_position(T_vectors):
    # Extract the translation components from each matrix (assuming they are in the last column)
    camera_positions = [T_vectors[i,:] for i in range(T_vectors.shape[0])]
    # Calculate the average position
    average_position = np.mean(camera_positions, axis=0)
    
    return average_position


def rotate_camera_matrix(R, t, axis_center, axis_direction, angle_degrees):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Normalize the axis direction vector
    axis_direction = axis_direction / np.linalg.norm(axis_direction)

    # Extract rotation (R) and translation (t) components from the world-to-camera matrix
    # R = world_to_cam[:, :3]
    # t = world_to_cam[:, 3]

    # Rodrigues' rotation formula components
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    one_minus_cos = 1.0 - cos_angle
    cross_matrix = np.array([
        [0, -axis_direction[2], axis_direction[1]],
        [axis_direction[2], 0, -axis_direction[0]],
        [-axis_direction[1], axis_direction[0], 0]
    ])

    # Rotation matrix for the axis
    axis_rotation_matrix = cos_angle * np.eye(3) + sin_angle * cross_matrix + one_minus_cos * np.outer(axis_direction, axis_direction)

    # Rotate the camera
    R_new = np.dot(axis_rotation_matrix, R)
    
    # Adjust translation component
    t_new = t - np.dot(axis_rotation_matrix, axis_center) + axis_center

    # Combine into new world-to-camera matrix
    new_world_to_cam = np.hstack((R_new, t_new.reshape(-1, 1)))

    return R_new, t_new