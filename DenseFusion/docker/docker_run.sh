#!/bin/bash
#
# Usage:  ./docker_run.sh [/path/to/data]
#
# This script calls `nvidia-docker run` to start the labelfusion
# container with an interactive bash session.  This script sets
# the required environment variables and mounts the labelfusion
# source directory as a volume in the docker container.  If the
# path to a data directory is given then the data directory is
# also mounted as a volume.
#

image_name=hoangcuongbk80/densefusion-gpu:latest

nvidia-docker run --name densefusion -it --rm -v /home/aass/Hoang-Cuong/Seg_SLAM/DenseFusion:/densefusion -v /media/aass/783de628-b7ff-4217-8c96-7f3764de70d9/Warehouse_Dataset:/warehouse_dataset densefusion-gpu /bin/bash

nvidia-docker run --name densefusion -it --rm -v /home/cghg/DenseFusion:/densefusion -v /media/DiskStation/trsv/data/YCB_Video_Dataset:/YCB_Video_Dataset hoangcuongbk80/densefusion-pytorch-1.0 /bin/bash

