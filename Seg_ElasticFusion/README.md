# Seg-ElasticFusion #

# 1. What do I need to build it? #

## 1.1. Ubuntu ##

* Ubuntu 14.04, 15.04 or 16.04 (Though many other linux distros will work fine)
* CMake
* OpenGL
* [CUDA >= 7.0](https://developer.nvidia.com/cuda-downloads)
* [OpenNI2](https://github.com/occipital/OpenNI2)
* SuiteSparse
* Eigen
* zlib
* libjpeg
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)
* [librealsense] (https://github.com/IntelRealSense/librealsense) - Optional (for Intel RealSense cameras)

Firstly, add [nVidia's official CUDA repository](https://developer.nvidia.com/cuda-downloads) to your apt sources, then run the following command to pull in most dependencies from the official repos:

```bash
sudo apt-get install -y cmake-qt-gui git build-essential libusb-1.0-0-dev libudev-dev openjdk-7-jdk freeglut3-dev libglew-dev cuda-7-5 libsuitesparse-dev libeigen3-dev zlib1g-dev libjpeg-dev
```

Afterwards install [OpenNI2](https://github.com/occipital/OpenNI2) and [Pangolin](https://github.com/stevenlovegrove/Pangolin) from source. Note, you may need to manually tell CMake where OpenNI2 is since Occipital's fork does not have an install option. It is important to build Pangolin last so that it can find some of the libraries it has optional dependencies on. 

When you have all of the dependencies installed, build the Core followed by the GUI. 

## 1.2. Windows - Visual Studio ##
* Windows 7/10 with Visual Studio 2013 Update 5 (Though other configurations may work)
* [CMake] (https://cmake.org/)
* OpenGL
* [CUDA >= 7.0](https://developer.nvidia.com/cuda-downloads)
* [OpenNI2](https://github.com/occipital/OpenNI2)
* [SuiteSparse] (https://github.com/jlblancoc/suitesparse-metis-for-windows)
* [Eigen] (http://eigen.tuxfamily.org)
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)
  * zlib (Pangolin can automatically download and build this)
  * libjpeg (Pangolin can automatically download and build this)
* [librealsense] (https://github.com/IntelRealSense/librealsense) - Optional (for Intel RealSense cameras)

Firstly install cmake and cuda. Then download and build from source OpenNI2, SuiteSparse. Next download Eigen (no need to build it since it is a header-only library). Then download and build from source Pangolin but pay attention to the following cmake settings. There will be a lot of dependencies where path was not found. That is OK except OPENNI2 and EIGEN3 (those should be set to valid paths). You also need to set MSVC_USE_STATIC_CRT to false in order to correctly link to ElasticFusion projects. Also, you can set BUILD_EXAMPLES to false since we don't need them and some were crashing on my machine.

Finally, build Core and GUI.


# 2. Is there an easier way to build it? #
Yes, if you run the *build.sh* script on a fresh clean install of Ubuntu 14.04, 15.04, or 16.04, enter your password for sudo a few times and wait a few minutes all dependencies will get downloaded and installed and it should build everything correctly. This has not been tested on anything but fresh installs, so I would advise using it with caution if you already have some of the dependencies installed.

# 3. Installation issues #

***`#include <Eigen/Core>` not found***

```bash
sudo ln -sf /usr/include/eigen3/Eigen /usr/include/Eigen
sudo ln -sf /usr/include/eigen3/unsupported /usr/include/unsupported
```

***invalid use of incomplete type ‘const struct Eigen ...***

Pangolin must be installed AFTER all the other libraries to make use of optional dependencies.

***GLSL 3.30 is not supported. Supported versions are 1.10, 1.20, 1.30, 1.00 ES and 3.00 ES***

Make sure you are running ElasticFusion on your nVidia GPU. In particular, if you have an Optimus GPU
- If you use Prime, follow instructions [here](http://askubuntu.com/questions/661922/how-am-i-supposed-to-use-nvidia-prime)
- If you use Bumblebee, remember to run as `optirun ./ElasticFusion`

# 4. How do I use it? #

Build core:
```bash
cd Core/
mkdir build
cd build
cmake ../src
make -j8
```

Build Seg_ElasticFusion:
```bash
cd offline-run/
mkdir build
cd build/
cmake ../src
make -j8
```

Run:
```bash
./Seg-ElasticFusion -l /home/aass/catkin_ws/src/Object-RPE/data/
```