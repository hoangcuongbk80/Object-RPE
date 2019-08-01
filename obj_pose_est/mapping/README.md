# 3D Mapping #
This is an implementation of 3D mapping in [Object-RPE](https://sites.google.com/view/object-rpe) based on the work of [ElasticFusion](https://github.com/mp3guy/ElasticFusion). 

## 1. What do I need to build it? #

* Ubuntu 16.04 (Though many other linux distros will work fine)
* CMake
* OpenGL
* [CUDA >= 7.0](https://developer.nvidia.com/cuda-downloads)
* OpenCV 3.1
* [OpenNI2](https://github.com/occipital/OpenNI2)
* SuiteSparse
* Eigen
* zlib
* libjpeg
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)

```bash
sudo apt-get install -y cmake-qt-gui git build-essential libusb-1.0-0-dev libudev-dev openjdk-7-jdk freeglut3-dev libglew-dev cuda-7-5 libsuitesparse-dev libeigen3-dev zlib1g-dev libjpeg-dev
```

Afterwards install [OpenNI2](https://github.com/occipital/OpenNI2) and [Pangolin](https://github.com/stevenlovegrove/Pangolin) from source. Note, you may need to manually tell CMake where OpenNI2 is since Occipital's fork does not have an install option. It is important to build Pangolin last so that it can find some of the libraries it has optional dependencies on. 

OpenNI2:
```bash
cd ~/catkin_ws/src/Object-RPE/obj_pose_est/mapping
mkdir deps
cd deps
git clone https://github.com/occipital/OpenNI2.git
cd OpenNI2
make -j8
```

Pangolin:
```bash
cd ~/catkin_ws/src/Object-RPE/obj_pose_est/mapping/deps
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON
make -j8
cd ../..
```

Pangolin must be installed AFTER all the other libraries to make use of optional dependencies.

When you have all of the dependencies installed, build the Core followed by the app.

## 2. Build core and app

Build core:
```bash
cd Core/
mkdir build
cd build
cmake ../src
make -j8
```

Build mapping app:
```bash
cd app/
mkdir build
cd build/
cmake ../src
make -j8
```