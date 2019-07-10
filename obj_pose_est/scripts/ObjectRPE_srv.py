#!/usr/bin/env python

from obj_pose_est.srv import *
import rospy
import os

def handle_ObjectRPE(req):
    print "Mask-RCNN running ..."
    maskrcnn_dir = os.getcwd() + '/src/Object-RPE/Mask_RCNN/samples/warehouse/warehouse_inference.py'
    maskrcnn_model_dir = os.getcwd() + '/src/Object-RPE/Mask_RCNN/logs/mask_rcnn_warehouse_0060.h5'
    data_dir = os.getcwd() + '/src/Object-RPE/data/'
    #aa = os.popen('python3 ' + maskrcnn_dir + ' --weights=' + maskrcnn_model_dir + ' --video=' + data_dir).read()
    
    print "3D mapping running ..."
    mapping_dir = os.getcwd() + '/src/Object-RPE/obj_pose_est/mapping/app/build/mapping'   
    #aa = os.popen(mapping_dir + ' -l /home/aass/catkin_ws/src/Object-RPE/data/').read()

    print("DenseFusion running ...")
    densefusion_dir = os.getcwd() + '/src/Object-RPE/DenseFusion/experiments/scripts/'   
    aa = os.popen('sh ' + densefusion_dir + 'inference_warehouse.sh').read()
    return ObjectRPEResponse(req.a + req.b)

def ObjectRPE_server():
    rospy.init_node('ObjectRPE_server')
    s = rospy.Service('Seg_Reconst_PoseEst', ObjectRPE, handle_ObjectRPE)
    print "Ready to run ObjectRPE."
    rospy.spin()

if __name__ == "__main__":
    print('The current working directory:')
    print(os.getcwd())
    ObjectRPE_server()