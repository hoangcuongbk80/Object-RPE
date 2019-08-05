# Object-RPE
This is an implementation of [Object-RPE](https://sites.google.com/view/object-rpe)

## Installation and compile the source
The tools require full ROS installation. The installation assumes you have Ubuntu 16.04 LTS [ROS Kinetic]
1. Clone the repository
   ```bash
   $ https://github.com/hoangcuongbk80/Object-RPE.git
   ```
2. ROS
   ```bash
   $ cd ~/catkin_ws
   $ catkin_make install
   ```
3. Segmentation [here](https://github.com/hoangcuongbk80/Object-RPE/tree/master/Mask_RCNN)
4. 3D mapping [here](https://github.com/hoangcuongbk80/Object-RPE/tree/master/obj_pose_est/mapping)
5. 6D object pose estimation [here](https://github.com/hoangcuongbk80/Object-RPE/master/iliad/DenseFusion)

## How to operate the system?

   ```bash
   $ roscore
   $ rosrun obj_pose_est ObjectRPE_srv.py
   $ roslaunch obj_pose_est launch_rpe_cam.launch
   $ roslaunch openni2_launch openni2.launch
   ```