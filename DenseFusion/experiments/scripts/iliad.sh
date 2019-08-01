#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/iliad.py --dataset_root /home/aass/catkin_ws/src/Object-RPE/data/dataset/warehouse\
  --saved_root /home/aass/catkin_ws/src/Object-RPE/data\
  --model trained_checkpoints/warehouse/pose_model_54_0.019968815155302567.pth\
  --refine_model trained_checkpoints/warehouse/pose_refine_model_92_0.01590736784950592.pth