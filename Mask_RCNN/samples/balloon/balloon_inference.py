"""
Usage: 
    # Apply mask-rcnn to an image
    python3 balloon_inference.py --weights=/home/aass/Hoang-Cuong/Mask_RCNN/logs/balloon20190515T1505/mask_rcnn_balloon_0003.h5 
    --image=/home/aass/Hoang-Cuong/Mask_RCNN/samples/balloon/1.jpg
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

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1 #2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

def get_masks(image, mask, class_ids):

    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    #print ("image shape: ", image.shape[1], image.shape[0])

    instance_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    semantic_masks = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # cuong start
    if mask.shape[-1] > 0:
        mask_zero = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        for i in range(mask.shape[-1]):
            semantic_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)            
            semantic_mask_one = semantic_mask_one * class_ids[i]
            semantic_masks = np.where(mask[:, :, i], semantic_mask_one, semantic_masks).astype(np.uint8)
            instance_mask_one = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)            
            instance_mask_one = instance_mask_one * (i+1)
            instance_masks = np.where(mask[:, :, i], instance_mask_one, instance_masks).astype(np.uint8)           
    
    return semantic_masks, instance_masks

def detect_and_get_masks(model, image_path=None):
    assert image_path or video_path

    # Run model detection and generate the color splash effect
    print("Running on {}".format(args.image))
    # Read image
    image = skimage.io.imread(args.image)
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    # Color splash
    semantic_masks, instance_masks = get_masks(image, r['masks'], r['class_ids'])

    plt.subplot(1, 2, 1)
    plt.title('rgb')
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title('masks')
    plt.imshow(instance_masks)
    plt.show()
    
    # Save output
    file_name = "mask_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, instance_masks)
    print("Saved to ", file_name)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect balloons.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()
    
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    class InferenceConfig(BalloonConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    weights_path = args.weights
    model.load_weights(weights_path, by_name=True)

    detect_and_get_masks(model, image_path=args.image)