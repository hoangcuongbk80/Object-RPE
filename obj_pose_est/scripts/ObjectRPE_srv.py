#!/usr/bin/env python

from obj_pose_est.srv import *
import rospy
import os

def handle_ObjectRPE(req):
    mask_dir = req.data_dir + '/mask/*'
    mask_color_dir = req.data_dir + '/mask-color/*'

    os.system('rm -r ' + mask_dir)
    os.system('rm -r ' + mask_color_dir)

    #--------------------------Start DenseFusion---------------------------
    print "Mask-RCNN running ..."

    executed_file = os.path.join(req.ObjectRPE_dir, 'Mask_RCNN/samples/warehouse/iliad.py') 
    maskrcnn_model_dir = ' --weights=' + os.path.join(req.data_dir, 'trained_models/warehouse/mask_rcnn.h5')
    num_frames = ' --num_frames=' + str(req.num_frames)
    num_keyframes = ' --num_keyframes=' + str(req.num_keyframes)
    data_dir = ' --data=' + req.data_dir
    aa = os.popen('python3 ' + executed_file  + maskrcnn_model_dir + data_dir + num_frames + num_keyframes).read()

    #--------------------------End DenseFusion---------------------------

    print "3D mapping running ..."
    executed_file = os.path.join(req.ObjectRPE_dir, 'obj_pose_est/mapping/app/build/mapping')
    data_dir = ' -l ' + req.data_dir + '/'
    num_frames = ' -num_frames ' + str(req.num_frames)
    aa = os.popen(executed_file + data_dir + num_frames).read()

    #--------------------------Start DenseFusion---------------------------
    print("DenseFusion running ...")

    densefusion_dir = req.ObjectRPE_dir + '/DenseFusion' 
    executed_file = ' ./tools/iliad.py'  

    dataset_root = ' --dataset_root ' + req.data_dir + '/dataset/warehouse'
    saved_root = ' --saved_root ' + req.data_dir
    pose_model = ' --model ' + os.path.join(req.data_dir, 'trained_models/warehouse/pose_model.pth')
    pose_refine_model = ' --refine_model ' + os.path.join(req.data_dir, 'trained_models/warehouse/pose_refine_model.pth')
    num_frames = ' --num_frames ' + str(req.num_frames)
    num_keyframes = ' --num_keyframes=' + str(req.num_keyframes)

    os.chdir(densefusion_dir)
    aa = os.popen('python3' + executed_file + dataset_root + saved_root + pose_model + pose_refine_model + num_frames + num_keyframes).read()
    #--------------------------End DenseFusion---------------------------
    
    return 1;

def ObjectRPE_server():
    rospy.init_node('ObjectRPE_server')
    s = rospy.Service('Seg_Reconst_PoseEst', ObjectRPE, handle_ObjectRPE)
    print "Ready to run ObjectRPE."
    rospy.spin()

if __name__ == "__main__":
    print('The current working directory:')
    print(os.getcwd())
    ObjectRPE_server()