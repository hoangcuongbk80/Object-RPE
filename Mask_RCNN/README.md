# Mask R-CNN for Object Detection and Segmentation

This is an implementation of instance segmentation used in [Object-RPE](https://sites.google.com/view/object-rpe). The original version is from [matterport](https://github.com/matterport/Mask_RCNN). The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

## Requirements
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages listed in `requirements.txt`.

## Installation
1. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
2. Run setup from the directory ~/catkin_ws/src/Object-RPE/Mask_RCNN
    ```bash
    sudo python3 setup.py install
    ```