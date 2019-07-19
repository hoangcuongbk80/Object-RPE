# Object-RPE
This is an implementation of [Object-RPE](https://sites.google.com/view/object-rpe)...

## Installation and compile the source

1. ROS
   ```bash
   $ cd ~/catkin_ws
   $ catkin_make install
   ```
2. Segmentation [here](https://github.com/hoangcuongbk80/Object-RPE/tree/master/Mask_RCNN)
3. 3D mapping [here](https://github.com/hoangcuongbk80/Object-RPE/tree/master/obj_pose_est/mapping)
4. 6D object pose estimation [here](https://github.com/hoangcuongbk80/Object-RPE/tree/master/DenseFusion)

## How do I use it?

```bash
   $ roscore
   $ rosrun obj_pose_est ObjectRPE_srv.py
   $ roslaunch obj_pose_est launch_object_rpe.launch
   $ roslaunch obj_pose_est launch_rpe_cam.launch
   ```