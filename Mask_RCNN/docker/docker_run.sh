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

image_name=hoangcuongbk80/maskrcnn-gpu:latest

sudo nvidia-docker run --name mask-rcnn -it --rm -v /home/aass/Hoang-Cuong/Mask_RCNN:/maskrcnn hoangcuongbk80/maskrcnn-gpu /bin/bash