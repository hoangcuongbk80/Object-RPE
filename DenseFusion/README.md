# DenseFusion
Based on the work of [Chen Wang](https://github.com/j96w/DenseFusion) <br />
Modified by Dinh-Cuong Hoang

## Requirements

* Python 3.5
* PyTorch 1.0
* torchvision 0.2.2.post3
* PIL
* scipy
* numpy
* pyyaml
* logging
* cffi
* matplotlib
* Cython
* CUDA 9.0/10.0

```bash
$ pip3 --no-cache-dir install numpy scipy pyyaml cffi pyyaml matplotlib Cython Pillow
$ pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1.post2-cp35-cp35m-linux_x86_64.whl
$ pip3 install torchvision==0.2.2.post3
```

## Train
1. To train
   ```bash
   sh /experiments/scripts/train_warehouse.sh
   ```
