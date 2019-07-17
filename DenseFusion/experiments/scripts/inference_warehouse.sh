#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/inference_warehouse.py --dataset_root /media/aass/783de628-b7ff-4217-8c96-7f3764de70d9/Warehouse_Dataset\
  --model trained_checkpoints/warehouse/pose_model_54_0.019968815155302567.pth\
  --refine_model trained_checkpoints/warehouse/pose_refine_model_92_0.01590736784950592.pth