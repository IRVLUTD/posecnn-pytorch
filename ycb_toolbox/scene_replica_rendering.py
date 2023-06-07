#!/usr/bin/env python

import _init_paths
import argparse
import os, sys
import torch
from transforms3d.quaternions import mat2quat, quat2mat, qmult
from transforms3d.euler import euler2quat, quat2euler
from fcn.config import cfg, cfg_from_file, get_output_dir
import scipy.io
import cv2
import numpy as np
import math
from utils.se3 import *
from ycb_renderer import YCBRenderer


classes_all = ('003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', \
               '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', \
               '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '037_scissors', '040_large_marker', \
               '052_extra_large_clamp')
               
class_colors_all = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), \
                    (0, 0, 128), (0, 128, 0), (128, 0, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128), \
                    (0, 64, 0), (64, 0, 0), (0, 0, 64), (64, 64, 0)]


def inverse_transform(trans):
    """
    Computes the inverse of 4x4 transform.

    Arguments:
        trans {np.ndarray} -- 4x4 transform.

    Returns:
        [np.ndarray] -- inverse 4x4 transform
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t
    return output


def convert_rospose_to_standard(pose_ros):
    """Converts (posn, x,y,z,w) pose to (posn, w,x,y,z) pose"""
    posn = pose_ros[:3]
    ros_qt = pose_ros[3:]
    quat = [ros_qt[-1], ros_qt[0], ros_qt[1], ros_qt[2]]
    return [*posn, *quat]


def convert_standard_to_rospose(pose_s):
    """Converts (posn, w,x,y,z) pose to ROS format (posn, x,y,z,w) pose"""
    posn = pose_s[:3]
    q_s = pose_s[3:]
    quat = [q_s[1], q_s[2], q_s[3], q_s[0]]
    return [*posn, *quat]


def ros_qt_to_rt(rot, trans):
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]
    qt[1] = rot[0]
    qt[2] = rot[1]
    qt[3] = rot[2]
    obj_T = np.eye(4)
    obj_T[:3, :3] = quat2mat(qt)
    obj_T[:3, 3] = trans

    return obj_T


def rt_to_ros_qt(rt):
    quat = mat2quat(rt[:3, :3])
    quat = [quat[1], quat[2], quat[3], quat[0]]
    trans = rt[:3, 3]

    return quat, trans


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='YCB rendering')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cad', dest='cad_name',
                        help='name of the CAD files',
                        default=None, type=str)
    parser.add_argument('--pose', dest='pose_name',
                        help='name of the pose files',
                        default=None, type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    root = '../data'
    height = 480
    width = 640
    
    # Setup data dir
    segdata_base = os.path.join(root, "final_scenes", "segmasks")
    os.makedirs(segdata_base, exist_ok=True)
    # Load SCENE IDS
    with open(os.path.join(root, "final_scenes", "scene_ids.txt"), "r") as f:
        selected_scene_ids = [int(x) for x in f.read().split()]

    # update camera intrinsics
    ROS_FETCH_K = [554.254691191187, 0.0, 320.5, 0.0, 554.254691191187, 240.5, 0.0, 0.0, 1.0]
    fx = ROS_FETCH_K[0]
    cx = ROS_FETCH_K[2]
    fy = ROS_FETCH_K[4]
    cy = ROS_FETCH_K[5]
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0,  0, 1]])
    # intrinsic_matrix = np.array([[500, 0, 320],
    #                              [0, 500, 240],
    #                              [0, 0, 1]])
    camera_pose = np.load(os.path.join(root, "final_scenes", "cam_pose.npy"))
    RT_base_to_cam = inverse_transform(camera_pose) # transform points in base frame to camera frame

    obj_paths = []
    texture_paths = []
    colors = []
    for cls_index in range(len(classes_all)):
        cls_name = classes_all[cls_index]
        obj_paths.append('{}/models/{}/textured_simple.obj'.format(root, cls_name))
        texture_paths.append('')
        colors.append(np.array(class_colors_all[cls_index]) / 255.0)
        
    print(obj_paths)
    print(texture_paths)
    print(colors)
    

    # setup renderer
    renderer = YCBRenderer(width=width, height=height, render_marker=False)
    renderer.load_objects(obj_paths, texture_paths, colors)

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    px = intrinsic_matrix[0, 2]
    py = intrinsic_matrix[1, 2]
    zfar = 10.0
    znear = 0.01

    renderer.set_camera_default()
    renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

    for scene_id in selected_scene_ids:
        print(f"SCENE: {scene_id}----------------------------------------------------------------------------")
        # Setup scene dir
        scene_dir = os.path.join(segdata_base, f"scene_{scene_id}")
        os.makedirs(scene_dir, exist_ok=True)

        # load object names and poses from scene metadata
        meta_fname = "meta-%06d.mat" % scene_id
        meta_fpath = os.path.join(root, "final_scenes", "metadata", meta_fname)
        meta = scipy.io.loadmat(meta_fpath)
        object_names = [o.strip() for o in meta['object_names']]
        poses = meta['poses']
        print(object_names)
        # Store poses in a dict
        meta_poses = {}
        for idx, obj in enumerate(object_names):
            meta_poses[obj] = meta["poses"][idx]

        # Get the object orders for this scene 
        random_order = str(meta["random"][0]).split(",")
        nearest_order = str(meta["nearest_first"][0]).split(",")

        for order in ["random", "nearest_first"]:
            ordering = random_order if order == "random" else nearest_order
            print(f"Scene {scene_id} | Order: {order}")
            print(f"Ordering: {ordering}")
            remain_objs = set(ordering)
            removed_objs = set()

            for step, obj_to_remove in enumerate(ordering):
                print(f"Step: {step} | Objects removed : {removed_objs}")
                print(f"Step: {step} | Objects in scene: {remain_objs}")
                # rendering to tensor
                image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
                seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

                # set object poses
                poses_all = []
                cls_indexes = []
                # for i in range(len(object_names)):
                for obj_in_scene in remain_objs:
                    # note, poses quat in meta are already in (wxyz) format
                    # qt = poses[i, :]
                    qt = meta_poses[obj_in_scene]
                    # 1. convert (q,t) to camera frame by multiplying the inverse 
                    # 2. convert (q,t) to standard format i.e wxyz format
                    qt_ros = convert_standard_to_rospose(qt)
                    RT_obj_base = ros_qt_to_rt(qt_ros[3:], qt_ros[:3])
                    RT_obj_cam = RT_base_to_cam @ RT_obj_base
                    q_cam, t_cam = rt_to_ros_qt(RT_obj_cam)
                    qt_cam_standard = convert_rospose_to_standard([*t_cam, *q_cam])
                    # Finally append to the list of all poses with correct tf
                    poses_all.append(qt_cam_standard)
                    index = classes_all.index(obj_in_scene)
                    cls_indexes.append(index)
                    
                renderer.set_poses(poses_all)
                renderer.set_light_pos([0, 0, 0])

                # rendering
                renderer.render(cls_indexes, image_tensor, seg_tensor)
                image_tensor = image_tensor.flip(0)
                seg_tensor = seg_tensor.flip(0)
                frame = [image_tensor.cpu().numpy(), seg_tensor.cpu().numpy()]

                im_syn = frame[0][:, :, :3] * 255
                im_syn = np.clip(im_syn, 0, 255)
                im_syn = np.around(im_syn).astype(np.uint8)

                im_label = frame[1][:, :, :3] * 255
                im_label = np.clip(im_label, 0, 255)
                im_label = np.around(im_label).astype(np.uint8)

                # save images
                import matplotlib.pyplot as plt
                rendersyn_fname = os.path.join(scene_dir, f"render_ord-{order}_step-{step}.png")
                plt.imsave(rendersyn_fname, im_syn)
                label_fname = os.path.join(scene_dir, f"gtseg_ord-{order}_step-{step}.png")
                plt.imsave(label_fname, im_label)

                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                plt.imshow(im_syn)
                # plt.plot(x1, y1, 'bo')
                ax.set_title('render')

                ax = fig.add_subplot(1, 2, 2)
                plt.imshow(im_label)
                ax.set_title('label')
                # plt.show()
                plot_fname = os.path.join(scene_dir, f"comparison_ord-{order}_step-{step}.png")
                plt.savefig(plot_fname)
                print(f"Step: {step} | Objects to remove: {obj_to_remove}")
                print("-------")
                remain_objs.remove(obj_to_remove)
                removed_objs.add(obj_to_remove)
            print("=========================")
        print("---------------------------------------------------------------------------------------------\n")

