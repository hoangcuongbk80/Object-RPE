"""
Mask R-CNN for Object_RPE
------------------------------------------------------------
python3 object_rpe.py --weights=/home/aass/Hoang-Cuong/Mask_RCNN/
logs/warehouse20190524T1156/mask_rcnn_warehouse_0060.h5 --image=1.png

python3 object_rpe.py --weights=/home/aass/Hoang-Cuong/Mask_RCNN/
logs/warehouse20190524T1156/mask_rcnn_warehouse_0060.h5 --video=/home/aass/Hoang-Cuong/datasets/warehouse_ECMR/0008/

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

def seq_get_masks(image, pre_detection, cur_detection):

    cur_masks = cur_detection['masks']
    cur_class_ids = cur_detection['class_ids']
    cur_rois = cur_detection['rois']

    pre_masks = pre_detection['masks']
    pre_class_ids = pre_detection['class_ids']
    pre_rois = pre_detection['rois']

    new_masks = pre_detection['masks']
    new_class_ids = pre_detection['class_ids']
    new_rois = pre_detection['rois']

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    semantic_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    semantic_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    good_detection = True
    print('cur_masks.shape: {}'.format(cur_masks.shape[-1]))

    if cur_masks.shape[-1] > 0:
        mask_zero = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i in range(cur_masks.shape[-1]):

            sub = np.abs(pre_rois-cur_rois[i])
            dist = sum(sub.transpose())
            print('cur_rois[i]: {}'.format(cur_rois[i]))
            mask_index = dist.argmin()
            if dist.min() < 50:
                if new_class_ids[mask_index] != cur_class_ids[i]: # bad classification
                    good_detection = False
                    break
                else:
                    new_rois[mask_index,:] = cur_rois[i,:] # change order of current masks to follow the mask order of previous prediction
            else:
                good_detection = False
                break

            semantic_mask = semantic_mask_one * cur_class_ids[i]
            semantic_masks = np.where(cur_masks[:, :, i], semantic_mask, semantic_masks).astype(np.uint8)

            instance_mask = instance_mask_one * (mask_index+1)
            instance_masks = np.where(cur_masks[:, :, i], instance_mask, instance_masks).astype(np.uint8)

    #print('old rois: \n {}'.format(cur_rois))
    #print('new rois: \n {}'.format(new_rois))

    instance_to_color = color_map()
    color_masks = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_masks[instance_masks == key] = instance_to_color[key]

    if good_detection:
        pre_detection['masks'] = new_masks
        pre_detection['class_ids'] = new_class_ids
        pre_detection['rois'] = new_rois
    return semantic_masks, instance_masks, color_masks, pre_detection, good_detection

def detect_and_get_masks(model, data_path, num_frames):
    assign_first_pre_detect = True
    for i in range(0, num_frames):
        num = 1000001 + i
        str_num = str(num)[1:]
        rgb_addr = "rgb/" + str_num + "-color.png"
        rgb_addr = os.path.join(data_path, rgb_addr)
        depth_addr = data_path + "/depth/" + str_num + "-depth.png"
        depth_addr = os.path.join(data_path, depth_addr)            

        # Read image
        image = skimage.io.imread(rgb_addr)
        depth = skimage.io.imread(depth_addr) 
        
        # Detect objects
        cur_detect = model.detect([image], verbose=1)[0]
        if assign_first_pre_detect and cur_detect['masks'].shape[-1] > 0:
            assign_first_pre_detect = False
            pre_detect = cur_detect
            file_dir = data_path + 'class_ids.txt' 
            with open(file_dir, 'w') as the_file:
                for j in range (cur_detect['class_ids'].shape[0]):
                    the_file.write(str(cur_detect['class_ids'][j]))
                    the_file.write('\n')
        # get instance_masks
        if not assign_first_pre_detect:
            semantic_masks, instance_masks, color_masks, pre_detect, good_detect  = seq_get_masks(image, pre_detect, cur_detect)

            if good_detect:
                mask_addr = data_path + '/mask-color/' + str_num + '.png'
                skimage.io.imsave(mask_addr, color_masks)
                mask_addr = data_path + '/mask/' + str_num + '.png'
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
    
    args = parser.parse_args()

    class InferenceConfig(WarehouseConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)
    
    detect_and_get_masks(model, args.data, args.num_frames)