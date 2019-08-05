#!/bin/bash
#
# Usage:  ./docker_run.sh [/path/to/data]
#
# This script calls `nvidia-docker run` to start the mask-rcnn
# container with an interactive bash session.  This script sets
# the required environment variables and mounts the labelfusion
# source directory as a volume in the docker container.  If the
# path to a data directory is given then the data directory is
# also mounted as a volume.
#

image_name=hoangcuongbk80/maskrcnn-gpu:latest

nvidia-docker run --name Object-RPE -it --rm -v /home/cghg/Object-RPE:/Object-RPE -v /media/DiskStation/trsv/data/YCB_Video_Dataset:/YCB_Video_Dataset hoangcuongbk80/maskrcnn-gpu /bin/bash

