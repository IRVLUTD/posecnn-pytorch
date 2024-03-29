#!/usr/bin/env python3

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Test a DeepIM network on an image database."""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import argparse
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import random
import scipy.io

import _init_paths
from fcn.test_dataset import test, test_autoencoder
from fcn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_dataset
import networks
from ycb_renderer import YCBRenderer
from fcn.pose_rbpf import PoseRBPF
from sdf.sdf_optimizer import sdf_optimizer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a PoseCNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--pretrained_encoder', dest='pretrained_encoder',
                        help='initialize with pretrained encoder checkpoint',
                        default=None, type=str)
    parser.add_argument('--codebook', dest='codebook',
                        help='codebook',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--dataset', dest='dataset_name',
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--dataset_background', dest='dataset_background_name',
                        help='background dataset to train on',
                        default='background_nvidia', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--background', dest='background_name',
                        help='name of the background file',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES
    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    print('GPU device {:d}'.format(args.gpu_id))

    # prepare dataset
    if cfg.TEST.VISUALIZE:
        shuffle = True
        np.random.seed()
    else:
        shuffle = False
    cfg.MODE = 'TEST'
    dataset = get_dataset(args.dataset_name)
    worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    if 'shapenet' in dataset.name:
        num_workers = 1
    else:
        num_workers = 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=shuffle,
        num_workers=num_workers, worker_init_fn=worker_init_fn)
    print('Use dataset `{:s}` for training'.format(dataset.name))

    if cfg.INPUT == 'COLOR':
        if cfg.TRAIN.SYN_BACKGROUND_SPECIFIC:
            background_dataset = get_dataset(args.dataset_background_name)
        else:
            background_dataset = get_dataset('background_coco')
    else:
        background_dataset = get_dataset('background_rgbd')
    background_loader = torch.utils.data.DataLoader(background_dataset, batch_size=cfg.TEST.IMS_PER_BATCH,
                                                    shuffle=True, num_workers=1)

    # overwrite intrinsics
    if len(cfg.INTRINSICS) > 0:
        K = np.array(cfg.INTRINSICS).reshape(3, 3)
        dataset._intrinsic_matrix = K
        background_dataset._intrinsic_matrix = K
        print(dataset._intrinsic_matrix)

    output_dir = get_output_dir(dataset, None)
    head, name = osp.split(args.pretrained)
    if cfg.PF.USE_DEPTH:
        suffix = '_depth' + '_particle_' + str(cfg.PF.N_PROCESS) + '_filter_' + str(cfg.PF.N_INIT_FILTERING)
    else:
        suffix = '_color' + '_particle_' + str(cfg.PF.N_PROCESS) + '_filter_' + str(cfg.PF.N_INIT_FILTERING)
    output_dir = osp.join(output_dir, name[:-15] + suffix)
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # prepare network
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        if isinstance(network_data, dict) and 'model' in network_data:
            network_data = network_data['model']
        print("=> using pre-trained network '{}'".format(args.pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[args.network_name](dataset.num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True

    print('loading 3D models')
    if 'shapenet' not in dataset.name:
        cfg.renderer = YCBRenderer(width=cfg.TRAIN.SYN_WIDTH, height=cfg.TRAIN.SYN_HEIGHT, gpu_id=args.gpu_id, render_marker=False)
        if cfg.TEST.SYNTHESIZE:
            cfg.renderer.load_objects(dataset.model_mesh_paths, dataset.model_texture_paths, dataset.model_colors)
        else:
            model_mesh_paths = [dataset.model_mesh_paths[i-1] for i in cfg.TEST.CLASSES]
            model_texture_paths = [dataset.model_texture_paths[i-1] for i in cfg.TEST.CLASSES]
            model_colors = [dataset.model_colors[i-1] for i in cfg.TEST.CLASSES]
            cfg.renderer.load_objects(model_mesh_paths, model_texture_paths, model_colors)
        cfg.renderer.set_camera_default()
        print(dataset.model_mesh_paths)

        if args.network_name == 'docsnet':
            dataset.compute_render_depths(cfg.renderer)

    # load sdfs
    if cfg.TEST.POSE_REFINE:
        print('loading SDFs')
        sdf_files = []
        for i in cfg.TEST.CLASSES[1:]:
            sdf_files.append(dataset.model_sdf_paths[i-1])
        cfg.sdf_optimizer = sdf_optimizer(cfg.TEST.CLASSES[1:], sdf_files)

    # test network
    if args.network_name == 'autoencoder':
        test_autoencoder(dataloader, background_loader, network, output_dir)
    elif args.network_name == 'docsnet':
        test_docsnet(dataloader, background_loader, network, output_dir)
    elif 'contrastive' in args.network_name:
        test_docsnet(dataloader, background_loader, network, output_dir, contrastive=True, prototype=False)
    elif 'prototype' in args.network_name:
        test_docsnet(dataloader, background_loader, network, output_dir, contrastive=False, prototype=True)
    elif 'triplet' in args.network_name:
        test_triplet_net(dataloader, background_loader, network, output_dir)
    elif 'rrn' in args.network_name:
        test_segnet(dataloader, background_loader, network, output_dir, rrn=True)
    elif 'seg' in args.network_name:
        test_segnet(dataloader, background_loader, network, output_dir, rrn=False)
    else:
        #'''
        # prepare autoencoder and codebook
        pose_rbpf = PoseRBPF(dataset, args.pretrained_encoder, args.codebook)
        test(dataloader, background_loader, network, pose_rbpf, output_dir)
        #'''

        # evaluation
        dataset.evaluation(output_dir)
