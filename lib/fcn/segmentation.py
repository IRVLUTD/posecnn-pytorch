import sys, os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
import imageio
import datasets
import utils.mask as util_
from pathlib import Path
from datasets.shapenet_utils import index_paths
from scipy.ndimage.measurements import label as connected_components
from time import time
from fcn.config import cfg
from networks import rrn_unet, rrn_vgg


class Segmentor(object):
    """ Class to encapsulate both Depth Seeding Network and Region Refinement Network

        There is NO training in this class
    """

    def __init__(self, params, rrn_filename):
        checkpoint = torch.load(rrn_filename)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            data = checkpoint['model']
        else:
            data = checkpoint

        if 'rrn_vgg' in rrn_filename:
            self.rrn = rrn_vgg(num_units=64, data=data)
        else:
            self.rrn = rrn_unet(num_units=64, data=data)
        self.rrn = torch.nn.DataParallel(self.rrn, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        self.rrn.eval()
        self.params = params
        self._pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float().cuda()
        self.textures_dir_dtd = Path(os.path.join(datasets.ROOT_DIR, 'data', 'textures'))
        self.texture_paths_dtd = index_paths(self.textures_dir_dtd, ext='.jpg')


    def load_random_image(self, paths):
        while True:
            image_path = random.choice(paths)
            try:
                image = imageio.imread(image_path)
                if len(image.shape) != 3 or image.shape[2] < 3:
                    continue
                return image[:, :, :3]
            except Exception:
                self._log.warning("failed to read image", path=image_path)


    def read_background(self):
        # read a background image
        background_color = self.load_random_image(self.texture_paths_dtd)
        bw = background_color.shape[1]
        bh = background_color.shape[0]
        x1 = np.random.randint(0, int(bw/3))
        y1 = np.random.randint(0, int(bh/3))
        x2 = np.random.randint(int(2*bw/3), bw)
        y2 = np.random.randint(int(2*bh/3), bh)
        background_color = background_color[y1:y2, x1:x2]
        background_color = cv2.resize(background_color, (cfg.TRAIN.SYN_CROP_SIZE, cfg.TRAIN.SYN_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
        if len(background_color.shape) != 3:
            background_color = cv2.cvtColor(background_color, cv2.COLOR_GRAY2RGB)
        background_color = torch.from_numpy(background_color).float() / 255.0
        background_color -= self._pixel_mean
        return background_color.permute(2, 0, 1)


    def process_initial_masks(self, initial_masks):
        """ Process the initial masks:
                - open/close morphological transform
                - closest connected component to object center
            @param initial_masks: a [N x H x W] torch.IntTensor. Note: Initial masks has values in [0, 2, 3, ...]. No table
        """
        N, H, W = initial_masks.shape

        # Bring some tensors to numpy for processing
        initial_masks = initial_masks.cpu().numpy()

        # Open/close morphology stuff
        if self.params['use_open_close_morphology']:

            for i in range(N):

                # Get object ids. Remove background (0)
                obj_ids = np.unique(initial_masks[i])
                if obj_ids[0] == 0:
                    obj_ids = obj_ids[1:]

                # For each object id, open/close the masks
                for obj_id in obj_ids:
                    mask = (initial_masks[i] == obj_id) # Shape: [H x W]
                    ksize = self.params['open_close_morphology_ksize'] # 9
                    opened_mask = cv2.morphologyEx(mask.astype(np.uint8), 
                                                   cv2.MORPH_OPEN, 
                                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

                    opened_closed_mask = cv2.morphologyEx(opened_mask,
                                                          cv2.MORPH_CLOSE,
                                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize,ksize)))

                    h_idx, w_idx = np.nonzero(mask)
                    initial_masks[i, h_idx, w_idx] = 0
                    h_idx, w_idx = np.nonzero(opened_closed_mask)
                    initial_masks[i, h_idx, w_idx] = obj_id

        # Closest Connected Component
        if self.params['use_closest_connected_component']:

            pixel_indices = util_.build_matrix_of_indices(H, W)
            for i in range(N):
                
                # Get object ids. Remove background (0)
                obj_ids = np.unique(initial_masks[i])
                if obj_ids[0] == 0:
                    obj_ids = obj_ids[1:]

                # Loop over each object
                for obj_index, obj_id in enumerate(obj_ids):

                    # Run connected components algorithm
                    components, num_components = connected_components(initial_masks[i] == obj_id)

                    # Find closest connected component via intersection over union
                    closest_component_num = -1
                    closest_component_iou = -1
                    mask_obj = initial_masks[i] == obj_id
                    for j in range(1, num_components+1):
                        mask = components == j
                        iou = np.sum((mask & mask_obj).astype(np.float32)) / np.sum((mask | mask_obj).astype(np.float32))
                        if iou > closest_component_iou:
                            closest_component_num = j
                            closest_component_iou = iou

                    # Fix the initial mask for this object
                    initial_masks[i][mask_obj] = 0
                    initial_masks[i][components == closest_component_num] = obj_id

        initial_masks = torch.from_numpy(initial_masks).to(cfg.device)
        return initial_masks


    def refine(self, rgb, initial_masks):
        """ Run algorithm on batch of images in eval mode

            @param batch: a dictionary with the following keys:
                            - rgb: a [N x 3 x H x W] torch.FloatTensor
                            - xyz: a [N x 3 x H x W] torch.FloatTensor
            @param final_close_morphology: If True, then run open/close morphology after refining mask.
                                           This typically helps a synthetically-trained RRN
        """
        # initial_masks: a [N x H x W] torch.IntTensor. Note: Initial masks has values in [0, 2, 3, ...]. No table
        N, _, H, W = rgb.shape
        initial_masks = self.process_initial_masks(initial_masks)
        crop_size = cfg.TRAIN.SYN_CROP_SIZE  # 224

        # Data structure to hold everything at end
        refined_masks = torch.zeros_like(initial_masks)
        for i in range(N):

            # Dictionary to save crop indices
            crop_indices = {}

            mask_ids = torch.unique(initial_masks[i])
            if mask_ids[0] == 0:
                mask_ids = mask_ids[1:]
            rgb_crops = torch.zeros((mask_ids.shape[0], 3, crop_size, crop_size), device=cfg.device)
            refined_crops = torch.zeros((mask_ids.shape[0], crop_size, crop_size), device=cfg.device)
            rois = torch.zeros((mask_ids.shape[0], 4), device=cfg.device)
            mask_crops = torch.zeros((mask_ids.shape[0], crop_size, crop_size), device=cfg.device)

            for index, mask_id in enumerate(mask_ids):
                mask = (initial_masks[i] == mask_id).float() # Shape: [H x W]

                # crop the masks/rgb to 224x224 with some padding, save it as "initial_masks"
                x_min, y_min, x_max, y_max = util_.mask_to_tight_box(mask)

                # Make bbox square
                cx = (x_min + x_max) / 2
                cy = (y_min + y_max) / 2
                x_delta = x_max - x_min
                y_delta = y_max - y_min
                if x_delta > y_delta:
                    y_min = cy - x_delta / 2
                    y_max = cy + x_delta / 2
                else:
                    x_min = cx - y_delta / 2
                    x_max = cx + y_delta / 2

                x_padding = int(torch.round((x_max - x_min).float() * self.params['padding_percentage']).item())
                y_padding = int(torch.round((y_max - y_min).float() * self.params['padding_percentage']).item())

                # Pad and be careful of boundaries
                x_min = max(x_min - x_padding, 0)
                x_max = min(x_max + x_padding, W-1)
                y_min = max(y_min - y_padding, 0)
                y_max = min(y_max + y_padding, H-1)
                crop_indices[mask_id.item()] = [x_min, y_min, x_max, y_max] # save crop indices
                rois[index, 0] = x_min
                rois[index, 1] = y_min
                rois[index, 2] = x_max
                rois[index, 3] = y_max

                # Crop
                rgb_crop = rgb[i, :, y_min:y_max+1, x_min:x_max+1] # [3 x crop_H x crop_W]
                mask_crop = mask[y_min:y_max+1, x_min:x_max+1] # [crop_H x crop_W]

                # Resize
                new_size = (crop_size, crop_size)
                rgb_crop = F.upsample_bilinear(rgb_crop.unsqueeze(0), new_size)[0] # Shape: [3 x new_H x new_W]
                rgb_crops[index] = rgb_crop
                mask_crop = F.upsample_nearest(mask_crop.unsqueeze(0).unsqueeze(0), new_size)[0,0] # Shape: [new_H, new_W]
                mask_crops[index] = mask_crop

            # Run the RGB Refinement Network
            if mask_ids.shape[0] > 0: # only run if you actually have masks to refine...
                inputs = torch.cat([rgb_crops, mask_crops.unsqueeze(1)], dim=1)
                label_blob = torch.cuda.FloatTensor(rgb_crops.shape[0], 2, crop_size, crop_size)
                refined_crops = self.rrn(inputs, label_blob) # Shape: [num_masks x new_H x new_W]

            # resize the results to the original size
            sorted_mask_ids = []
            for index, mask_id in enumerate(mask_ids):

                # Resize back to original size
                x_min, y_min, x_max, y_max = crop_indices[mask_id.item()]
                orig_H = y_max - y_min + 1
                orig_W = x_max - x_min + 1
                mask = refined_crops[index].unsqueeze(0).unsqueeze(0).float()
                resized_mask = F.upsample_nearest(mask, (orig_H, orig_W))[0,0]

                # mask rois
                background = -1 * (1 - refined_crops[index].unsqueeze(2)) * self._pixel_mean
                rgb_crops[index] = refined_crops[index] * rgb_crops[index] + background.permute(2, 0, 1)

                # Set refined mask
                h_idx, w_idx = torch.nonzero(resized_mask).t()
                refined_masks[i, y_min:y_max+1, x_min:x_max+1][h_idx, w_idx] = mask_id

        return refined_masks, refined_crops, rgb_crops, rois
