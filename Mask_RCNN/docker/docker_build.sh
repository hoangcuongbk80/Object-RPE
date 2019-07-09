#!/bin/bash
#
# This script runs docker build to create the maskrcnn-gpu docker image.
#

set -exu
sudo nvidia-docker build --tag maskrcnn-gpu ./