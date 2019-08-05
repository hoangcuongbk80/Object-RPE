"""
Mask R-CNN
Based on the work of Waleed Abdulla (Matterport)
Modified by Dinh-Cuong Hoang
------------------------------------------------------------
python3 eval.py
It will read rgb and ground-truth images from /rgb and gt/ folders in .../Object-RPE/data
then save results in .../Object-RPE/data/mask
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class WarehouseConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Warehouse"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 400

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

def color_map():
    color_map_dic = {
    0:  [0, 0, 0],
    1:  [128, 128,   0],
    2:  [  0, 128, 128],
    3:  [128,   0, 128],
    4:  [128,   0,   0], 
    5:  [  0, 128,   0],
    6:  [  0,   0, 128],
    7:  [255, 255,   0],
    8:  [255,   0, 255],
    9:  [  0, 255, 255],
    10: [255,   0,   0],
    11: [  0, 255,   0],
    12: [  0,   0, 255],
    13: [ 92,  112, 92],
    14: [  0,   0,  70],
    15: [  0,  60, 100],
    16: [  0,  80, 100],
    17: [  0,   0, 230],
    18: [119,  11,  32],
    19: [  0,   0, 121]
    }
    return color_map_dic

def get_masks(image, mask, class_ids):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    semantic_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    if mask.shape[-1] > 0:
        mask_zero = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i in range(mask.shape[-1]):
            semantic_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)            
            semantic_mask_one = semantic_mask_one * class_ids[i]
            semantic_masks = np.where(mask[:, :, i], semantic_mask_one, semantic_masks).astype(np.uint8)
            instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)            
            instance_mask_one = instance_mask_one * (i+1)
            instance_masks = np.where(mask[:, :, i], instance_mask_one, instance_masks).astype(np.uint8)           
    
    instance_to_color = color_map()
    color_masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_masks[instance_masks == key] = instance_to_color[key]
    
    return semantic_masks, instance_masks, color_masks

def detect_and_get_masks(model):
    data_dir = os.path.abspath("../../../") + '/data/'
    rgb_dir = data_dir + 'rgb/'
    mask_dir = data_dir + 'mask/'
    gt_dir = data_dir + 'gt/'
    rgb_addrs = glob.glob(rgb_dir + '*.png')
    global_accuracy = 0
    accur_dir = data_dir + 'mask/accuracy.txt'
    accur_file = open(accur_dir, 'w')
    count = 0

    for i in range(len(rgb_addrs)):
        str_num = rgb_addrs[i][len(rgb_dir):len(rgb_dir)+6]
        # Read image
        image = skimage.io.imread(rgb_addrs[i])
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # get instance_masks
        semantic_masks, instance_masks, color_masks  = get_masks(image, r['masks'], r['class_ids'])
        # save mask image
        mask_addr = data_dir + 'mask/' + rgb_addrs[i][len(rgb_dir):]
        skimage.io.imsave(mask_addr, color_masks)
        # evaluation
        gt_addr = gt_dir + str_num + '-label.png' 
        if os.path.isfile(gt_addr) == False: 
            continue;
        gt = skimage.io.imread(gt_addr)
        dif = semantic_masks - gt
        accuracy = 1 - np.count_nonzero(dif) / (gt.shape[0]*gt.shape[1])
        accur_file.write('%s %f\n' % (str_num, accuracy))
        global_accuracy = global_accuracy + accuracy
        count +=1
    if count:
        accur_file.write('Mean accuracy: %f' % (global_accuracy / count))
    accur_file.close()
        
    
if __name__ == '__main__':

    class InferenceConfig(WarehouseConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    weights_dir = os.path.abspath("../../../") + '/data/trained_models/warehouse/mask_rcnn.h5'
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=weights_dir)
    model.load_weights(weights_dir, by_name=True)
    
    detect_and_get_masks(model)