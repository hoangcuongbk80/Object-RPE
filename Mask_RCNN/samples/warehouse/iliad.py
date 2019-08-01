"""
Mask R-CNN for Object_RPE
------------------------------------------------------------
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
    DETECTION_MIN_CONFIDENCE = 0.7

def seq_get_masks(image, cur_detection):

    cur_masks = cur_detection['masks']
    cur_class_ids = cur_detection['class_ids']
    cur_rois = cur_detection['rois']

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    semantic_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    semantic_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    print('cur_masks.shape: {}'.format(cur_masks.shape[-1]))

    if cur_masks.shape[-1] > 0:
        mask_zero = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i in range(cur_masks.shape[-1]):
            semantic_mask = semantic_mask_one * cur_class_ids[i]
            semantic_masks = np.where(cur_masks[:, :, i], semantic_mask, semantic_masks).astype(np.uint8)

            instance_mask = instance_mask_one * (i+1)
            instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

    return semantic_masks, instance_masks

def detect_and_get_masks(model, data_path, num_frames, num_keyframes):
    incr = num_frames//num_keyframes
    if incr < 1:
        return;

    for i in range(0, num_frames, incr):
        num = 1000001 + i
        str_num = str(num)[1:]
        rgb_addr = "rgb/" + str_num + "-color.png"
        rgb_addr = os.path.join(data_path, rgb_addr)
        depth_addr = "depth/" + str_num + "-depth.png"
        depth_addr = os.path.join(data_path, depth_addr)         

        if os.path.isfile(rgb_addr) == False: 
            continue;
        if os.path.isfile(depth_addr) == False: 
            continue;

        # Read image
        image = skimage.io.imread(rgb_addr)
        depth = skimage.io.imread(depth_addr)
        
        # Detect objects
        cur_detect = model.detect([image], verbose=1)[0]
            
        file_name = 'mask/' + str_num + '-class_ids.txt'
        file_dir = os.path.join(data_path, file_name)
        with open(file_dir, 'w') as the_file:
            for j in range (cur_detect['class_ids'].shape[0]):
                the_file.write(str(cur_detect['class_ids'][j]))
                the_file.write('\n')
        
        # get instance_masks
        semantic_masks, instance_masks  = seq_get_masks(image, cur_detect)
        
        file_name = 'mask/' + str_num + '-mask.png'
        mask_addr = os.path.join(data_path, file_name) 
        skimage.io.imsave(mask_addr, instance_masks)
        
        plt.subplot(2, 2, 1)
        plt.title('rgb')
        plt.imshow(image)
        plt.subplot(2, 2, 2)
        plt.title('depth')
        plt.imshow(depth)
        plt.subplot(2, 2, 3)
        plt.title('mask')
        plt.imshow(instance_masks)
        plt.subplot(2, 2, 4)
        plt.title('label')
        plt.imshow(semantic_masks)
        plt.draw()
        plt.pause(0.001)
    #plt.show()

    
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
                        description='Train Mask R-CNN to detect Warehouses.')
    parser.add_argument('--data', required=False,
                        metavar="/path/to/data/",
                        help='Directory of the Warehouse dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--num_frames', type=int, default = 100, help='number of images')
    parser.add_argument('--num_keyframes', type=int, default = 10, help='real number of images applied')
    
    args = parser.parse_args()

    class InferenceConfig(WarehouseConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)
    
    detect_and_get_masks(model, args.data, args.num_frames, args.num_keyframes)