# DenseFusion

This is an implementation of 6D object estimation used in [Object-RPE](https://sites.google.com/view/object-rpe). The original version is from [DenseFusion](https://github.com/j96w/DenseFusion).

## Requirements

* Python 2.7/3.5/3.6 (If you want to use Python2.7 to run this repo, please rebuild the `lib/knn/` (with PyTorch 0.4.1).)
* [PyTorch 0.4.1](https://pytorch.org/) ([PyTroch 1.0 branch](<https://github.com/j96w/DenseFusion/tree/Pytorch-1.0>))
* PIL
* scipy
* numpy
* pyyaml
* logging
* matplotlib
* CUDA 7.5/8.0/9.0 (Required. CPU-only will lead to extreme slow training speed because of the loss calculation of the symmetry objects (pixel-wise nearest neighbour loss).)

## Training

* YCB_Video Dataset:
	After you have downloaded and unzipped the YCB_Video_Dataset.zip and installed all the dependency packages, please run:
```	
./experiments/scripts/train_ycb.sh
```
* Warehouse Dataset:
```	
./experiments/scripts/train_warehouse.sh
```
**Training Process**: The training process contains two components: (i) Training of the DenseFusion model. (ii) Training of the Iterative Refinement model. In this code, a DenseFusion model will be trained first. When the average testing distance result (ADD for non-symmetry objects, ADD-S for symmetry objects) is smaller than a certain margin, the training of the Iterative Refinement model will start automatically and the DenseFusion model will then be fixed. You can change this margin to have better DenseFusion result without refinement but it's inferior than the final result after the iterative refinement. 

**Checkpoints and Resuming**: After the training of each 1000 batches, a `pose_model_current.pth` / `pose_refine_model_current.pth` checkpoint will be saved. You can use it to resume the training. After each testing epoch, if the average distance result is the best so far, a `pose_model_(epoch)_(best_score).pth` /  `pose_model_refiner_(epoch)_(best_score).pth` checkpoint will be saved. You can use it for the evaluation.

**Notice**: The training of the iterative refinement model takes some time. Please be patient and the improvement will come after about 30 epoches.