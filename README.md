# 3D Reconstruction and 6D Object Pose Estimation
This contains a brief guide how to install / run the ROS-based [Object-RPE](https://sites.google.com/view/object-rpe) developed at ORU in D6.4 [ILIAD](https://iliad-project.eu) project.

![The set of 11 objects](figs/ex1.png)

## Installation and compile the source

1. ROS
   ```bash
   $ cd ~/catkin_ws
   $ catkin_make install
   ```
2. Segmentation [here](https://github.com/hoangcuongbk80/Object-RPE/tree/master/Mask_RCNN)
3. 3D mapping [here](https://github.com/hoangcuongbk80/Object-RPE/tree/master/obj_pose_est/mapping)
4. 6D object pose estimation [here](https://github.com/hoangcuongbk80/Object-RPE/tree/master/DenseFusion)

## Testing

1. Download data folder here and copy to ~/catkin_ws/src/Object-RPE
2. To test segmenation module:
   ```bash
   $ cd ~/catkin_ws/src/Object_RPE/Mask-RCNN/samples/warehouse
   $ python3
   ```
3. To test 3D mapping module:
   ```bash
   $ cd ~/catkin_ws/src/Object-RPE/obj_pose_est/mapping/app
   $ python3
   ```
4. To test pose estimation module:
   ```bash
   $ cd ~/catkin_ws/src/Object-RPE/obj_pose_est/mapping/app
   $ python3
   ```

## How to operate the system:

```bash
   $ roscore
   $ rosrun obj_pose_est ObjectRPE_srv.py
   $ roslaunch obj_pose_est launch_object_rpe.launch
   $ roslaunch obj_pose_est launch_rpe_cam.launch
   ```