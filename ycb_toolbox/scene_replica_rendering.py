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
    
    # update camera intrinsics
    intrinsic_matrix = np.array([[500, 0, 320],
                                 [0, 500, 240],
                                 [0, 0, 1]])

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
    
    # load object names and poses from scene metadata
    scene_id = 1
    filename = '%s/final_scenes/metadata/meta-%06d.mat' % (root, scene_id)
    print(filename)
    meta = scipy.io.loadmat(filename)
    object_names = meta['object_names']
    poses = meta['poses']
    print(object_names)
    print(poses, poses.shape)

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

    # rendering to tensor
    image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
    seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()

    # set object poses
    poses_all = []
    cls_indexes = []
    for i in len(object_names):        
        qt = poses[i, :]
        poses_all.append(qt)
        
        index = classes_all.index(object_names[i])
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
    im_syn = im_syn.astype(np.uint8)

    im_label = frame[1][:, :, :3] * 255
    im_label = np.clip(im_label, 0, 255)
    im_label = im_label.astype(np.uint8)

    # show images
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(im_syn)
    plt.plot(x1, y1, 'bo')
    ax.set_title('render')

    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(im_label)
    ax.set_title('label')
    plt.show()
