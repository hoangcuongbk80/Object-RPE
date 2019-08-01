#include <iostream> 
#include <cstdlib> 
#include <stdio.h>
#include <stdlib.h>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>

#include "obj_pose_est/model_to_scene.h"
#include "obj_pose_est/ObjectRPE.h"

using namespace std;
using namespace cv;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr  scene_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_scene (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr  pub_one_pred (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr  pub_multi_pred (new pcl::PointCloud<pcl::PointXYZRGB>);

std::string depth_path, rgb_path;
cv::Mat rgb_img, depth_img;
double fx, fy, cx, cy, depth_factor;
double dst_thresh; // minimum distance between two points considered as overlapped
double overlap_thresh; // minimum overlap portion between two poinclouds considered as the same instance
double confidence_thresh; // minimum confidence score of pose estimation for picking

std::vector<string> model_paths; // path to model of object detected
std::vector<string> full_items; // full list of item names in dataset
std::vector<Eigen::Matrix4f> transforms;
Eigen::Matrix4f cam_T;

std::vector<pcl::PointCloud<pcl::PointXYZRGB>> curr_models;
std::vector<string> curr_objects; // names of object detected in the current image
std::vector<int> curr_clsIDs; // class ID of object detected in the current image

std::vector<pcl::PointCloud<pcl::PointXYZRGB>> global_models;
std::vector<string> global_objects; // names of object detected in the current image
std::vector<int> global_clsIDs; // class ID of object detected in the current image
std::vector<Eigen::Matrix4f> global_transforms;
std::vector<double> confidence_scores;

bool depthToClould()
{
  depth_img = cv::imread(depth_path, -1);
  rgb_img = cv::imread(rgb_path, -1);

  if(!rgb_img.data || !depth_img.data)
  {
      if(!rgb_img.data) std::cerr << "Cannot read image from " << rgb_path << "\n"; 
      else std::cerr << "Cannot read image from " << depth_path << "\n";
      return false;
  }


  cv::imshow("rgb", rgb_img);
  cv::waitKey(300);

   scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
   pcl::PointXYZRGB point;
   for(int row=0; row < depth_img.rows; row++)
    {
       for(int col=0; col < depth_img.cols; col++)       
        {
          if(isnan(depth_img.at<ushort>(row, col))) continue;
          double depth = depth_img.at<ushort>(row, col) / depth_factor;
          point.x = (col-cx) * depth / fx;
          point.y = (row-cy) * depth / fy;
          point.z = depth;
          point.b = rgb_img.at<cv::Vec3b>(row, col)[0];
          point.g = rgb_img.at<cv::Vec3b>(row, col)[1];
          point.r = rgb_img.at<cv::Vec3b>(row, col)[2];
         scene_cloud->push_back(point);
        }
    }
    return true;
}

void colorMap(int i, pcl::PointXYZRGB &point)
{
  if (i == 1) // red  
  {
    point.r = 255; point.g = 0; point.b = 0; 
  }
  else if (i == 2) //line
  {
    point.r = 0; point.g = 255; point.b = 0;
  }
  else if ( i == 3) //blue
  { 
    point.r = 0; point.g = 0; point.b = 255;
  } 
  else if ( i == 4) //maroon
  {
    point.r = 128; point.g = 0; point.b = 0;

  }
  else if ( i == 5) //green
  {
    point.r = 0; point.g = 128; point.b = 0;
  }  
  else if ( i == 6) //navy
  {
    point.r = 0; point.g = 0; point.b = 128;
  }
  else if ( i == 7) //yellow
  {
    point.r = 255; point.g = 255; point.b = 0;
  }
  else if ( i == 8) //magenta
  {
    point.r = 255; point.g = 0; point.b = 255;
  }
  else if ( i == 9) //cyan
  {
    point.r = 0; point.g = 255; point.b = 255;
  }    
  else if ( i == 10) //olive
  {
    point.r = 128; point.g = 128; point.b = 0;
  }
  else if ( i == 11) //purple
  {
    point.r = 128; point.g = 0; point.b = 128;
  } 
    
  else if ( i == 12) //teal
  {
    point.r = 0; point.g = 128; point.b = 128;
  }
    
  else if ( i == 13) 
  {
    point.r = 92; point.g = 112; point.b = 92;
  }
  else if ( i == 14) //brown
  {
    point.r = 165; point.g = 42; point.b = 42;
  }    
  else //silver
  {
    point.r = 192; point.g = 192; point.b = 192;
  }                   
}

double overlapPortion(const pcl::PointCloud<pcl::PointXYZRGB> &source, 
                                          const pcl::PointCloud<pcl::PointXYZRGB> &target, 
                                          const double &max_dist)
{
	if (source.size() == 0 || target.size() == 0) return -1;
	pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(target, *target_cloud);
	pcl::copyPointCloud(source, *source_cloud);
	
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(target_cloud);
	std::vector<int> pointIdxNKNSearch(1);
	std::vector<float> pointNKNSquaredDistance(1);

	int overlap_Points = 0;
	for (int i = 0; i < source.size(); ++i)
	{
		if (kdtree.nearestKSearch(source_cloud->points[i], 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
            if(sqrt(pointNKNSquaredDistance[0]) < max_dist)
			    overlap_Points++;
		}
	}

	//calculating the mean distance
	double portion = (double) overlap_Points / source.size();
	return portion;
}

void processModels()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr  model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>> original_models;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr  color_model_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  
  for(int i=0; i < model_paths.size(); i++)
  {
    pcl::io::loadPLYFile<pcl::PointXYZ> (model_paths[i], *model_cloud);
    copyPointCloud(*model_cloud, *color_model_cloud);
    original_models.push_back(*color_model_cloud);
    pcl::transformPointCloud(*color_model_cloud, *color_model_cloud, transforms[i]);
    for(int k=0; k < color_model_cloud->size(); k++)
    {
      colorMap(i+1, color_model_cloud->points[k]);
    }    
    curr_models.push_back(*color_model_cloud);
  }
  
  if(!curr_models.size()) return;
  if(!global_models.size())
  {
    for(int i=0; i < curr_models.size(); i++)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr  transformed_model (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::transformPointCloud(curr_models[i], *transformed_model, cam_T);
      global_models.push_back(*transformed_model);
      global_objects.push_back(curr_objects[i]);
      global_clsIDs.push_back(curr_clsIDs[i]);
      confidence_scores.push_back(1);
      Eigen::Matrix4f global_T = cam_T * transforms[i];
      global_transforms.push_back(global_T);
    }
    return;
  }

  for(int i=0; i < curr_models.size(); i++)
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  transformed_model (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(curr_models[i], *transformed_model, cam_T);
    int numOfGlobalObjects = global_models.size();
    bool new_object = true;
    
    for(int j=0; j < numOfGlobalObjects; j++)
    {
      if(curr_clsIDs[i] == global_clsIDs[j])
      {
        double overlap = overlapPortion(*transformed_model, global_models[i], dst_thresh);
        if(overlap > overlap_thresh)
        {
          Eigen::Matrix4f global_T = cam_T * transforms[i];
          global_T = (global_transforms[j] * confidence_scores[i] + global_T) / (confidence_scores[i]+1);
          pcl::transformPointCloud(original_models[i], *transformed_model, global_T);

          for(int k=0; k < transformed_model->size(); k++)
          {
            colorMap(j+1, transformed_model->points[k]);
          }
          global_models[j] = *transformed_model;
          confidence_scores[j] = confidence_scores[i] * (1+overlap);
        }
        new_object = false;
      }
    }

    if(new_object)
    {
      for(int k=0; k < transformed_model->size(); k++)
      {
        colorMap(global_models.size()+1, transformed_model->points[k]);
      }
      global_models.push_back(*transformed_model);
      global_clsIDs.push_back(curr_clsIDs[i]);
      global_objects.push_back(curr_objects[i]);
      confidence_scores.push_back(1);
      Eigen::Matrix4f global_T = cam_T * transforms[i];
      global_transforms.push_back(global_T);
    }

  }
}

void getCalibrationParas(std::string dataset)
{
  if(dataset == "YCB-Video")
  {

  }
  if(dataset == "Warehouse")
  {
    fx=580.0; fy=580.0;
    cx=319.0; cy=237.0;
    depth_factor = 1000;
  }
}

void extract_transform_from_quaternion(std::string line, Eigen::Matrix4f &T, int class_index)
{
	  Eigen::Vector3f trans;
    float rot_quaternion[4];
    vector<string> st;
    boost::trim(line);
		boost::split(st, line, boost::is_any_of("\t\r "), boost::token_compress_on);
    trans(0) = std::stof(st[4]); trans(1) = std::stof(st[5]); trans(2) = std::stof(st[6]); //translaton
    rot_quaternion[0] = std::stof(st[0]); rot_quaternion[1] = std::stof(st[1]); //rotation
    rot_quaternion[2] = std::stof(st[2]); rot_quaternion[3] = std::stof(st[3]); //rotation

    Eigen::Quaternionf q(rot_quaternion[0], rot_quaternion[1], rot_quaternion[2], rot_quaternion[3]); //w x y z
    
    T.block(0, 3, 3, 1) = trans;
    T.block(0, 0, 3, 3) = q.normalized().toRotationMatrix();
}

void extract_cam_pose(std::string line)
{
	  Eigen::Vector3f trans;
    float rot_quaternion[4];
    vector<string> st;
    boost::trim(line);
		boost::split(st, line, boost::is_any_of("\t\r "), boost::token_compress_on);
    trans(0) = std::stof(st[1]); trans(1) = std::stof(st[2]); trans(2) = std::stof(st[3]); //translaton
    rot_quaternion[0] = std::stof(st[7]); rot_quaternion[1] = std::stof(st[4]); //rotation
    rot_quaternion[2] = std::stof(st[5]); rot_quaternion[3] = std::stof(st[6]); //rotation

    Eigen::Quaternionf q(rot_quaternion[0], rot_quaternion[1], rot_quaternion[2], rot_quaternion[3]); //w x y z
    
    cam_T.setIdentity();
    cam_T.block(0, 3, 3, 1) = trans;
    cam_T.block(0, 0, 3, 3) = q.normalized().toRotationMatrix();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ObjectRPE");

  ros::NodeHandle nh_, nh_srv, nh_cloud, cloud_mul_n;
  ros::Publisher cloud_pub_one_pred = nh_cloud.advertise<sensor_msgs::PointCloud2> ("one_pred", 1);
  ros::Publisher cloud_pub_multi_pred = cloud_mul_n.advertise<sensor_msgs::PointCloud2> ("multi_pred", 1);
  ros::Rate loop_rate(10);

  std::string dataset, ObjectRPE_dir, data_dir; 
  int num_frames, num_keyframes;
  bool call_service;

  nh_ = ros::NodeHandle("~");
  nh_.getParam("ObjectRPE_dir", ObjectRPE_dir);
  nh_.getParam("dataset", dataset);
  nh_.getParam("data_dir", data_dir);
  nh_.getParam("num_frames", num_frames);
  nh_.getParam("num_keyframes", num_keyframes);  
  nh_.getParam("call_service", call_service);  
  nh_.getParam("dst_thresh", dst_thresh);
  nh_.getParam("overlap_thresh", overlap_thresh);
  nh_.getParam("confidence_thresh", confidence_thresh);

  if(num_keyframes > num_frames) std::cerr << "The number of keyframes cannot larger than the number of frames!";
  getCalibrationParas(dataset);

  //-----------------------Call service for MaskRCNN and DenseFusion---------------------------

  if(call_service)
  {
    ros::ServiceClient client = nh_srv.serviceClient<obj_pose_est::ObjectRPE>("Seg_Reconst_PoseEst");
    obj_pose_est::ObjectRPE srv;

    srv.request.ObjectRPE_dir = ObjectRPE_dir;
    srv.request.dataset = dataset;
    srv.request.data_dir = data_dir;
    srv.request.num_frames = num_frames;
    srv.request.num_keyframes = num_keyframes;

    ROS_INFO("ObjectRPE running");
    
    if (client.call(srv))
    {
      ROS_INFO("Result: %ld", (long int)srv.response.ouput);
    }
    else
    {
      ROS_ERROR("Failed to call service ObjectRPE");
      return 1;
    }
  }
 
  //-----------------------------------Process predictions-------------------------------------
  
  std::string classes_path = data_dir + "/dataset/warehouse/image_sets/classes.txt";
  std::string model_dir = data_dir + "/dataset/warehouse/models";
  pcl::PCLPointCloud2 cloud_filtered;
  sensor_msgs::PointCloud2 output;
  
  ifstream classes_file (classes_path);
  if (classes_file.is_open())                     
    {
      while (!classes_file.eof())                 
      {
        string cls;
        getline (classes_file, cls);
        full_items.push_back(cls);
      }
    }
    else 
    {
      std::cerr << "Unable to open " << classes_path  << " file" << "\n";
      exit(0);
    }
  classes_file.close();

  std::string cam_traject_path = data_dir + "/map.freiburg";
  ifstream cam_traject_file (cam_traject_path);
  if(cam_traject_file.fail())
  {
    std::cerr << "Cannot read " << cam_traject_path << "n";
    return 1;
  }

  int now = 0;
  while (ros::ok())
  {
    if(now < num_frames) now++;
    else 
    {
      now = 0;
      pub_multi_pred.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
      cam_traject_file.clear();
      cam_traject_file.seekg(0, ios::beg);
      global_models.clear();
      global_objects.clear();
      global_clsIDs.clear();
      confidence_scores.clear();
      global_transforms.clear();
      continue;
    }

    string cam_line;
    if(!cam_traject_file.eof()) getline (cam_traject_file, cam_line);
    
    int num = 1000000 + now;
    std::string str_num = std::to_string(num).substr(1, 6);
    std::string pose_path = data_dir + "/mask/" + str_num + "-object_poses.txt";

    ifstream posefile (pose_path);
    if(posefile.fail()) continue;

    pub_one_pred.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    pub_multi_pred.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    transformed_scene.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    model_paths.clear();
    transforms.clear();
    curr_objects.clear();
    curr_clsIDs.clear();
    curr_models.clear();

    depth_path = data_dir + "/depth/" + str_num + "-depth.png";
    rgb_path = data_dir + "/rgb/" + str_num + "-color.png";            

    string line;

    if (posefile.is_open())            
    {
      while(!posefile.eof())
      {
        if(!posefile.eof()) getline (posefile, line);
        if(line=="") continue;
        curr_clsIDs.push_back(std::stoi(line));
        int cls_index = std::stoi(line) - 1;
        std::string model_path = model_dir + "/" + full_items[cls_index] + "/points.ply";
        
        curr_objects.push_back(full_items[cls_index]);
        model_paths.push_back(model_path);
        if(!posefile.eof()) getline (posefile, line);
        if(line=="") continue;

        Eigen::Matrix4f T(Eigen::Matrix4f::Identity());
        extract_transform_from_quaternion(line, T, cls_index);          
        transforms.push_back(T);
      }
    }
    else 
    {
      std::cerr << "Unable to open file";
      exit(0);
    }
    if(model_paths.size())
    {
      if(!depthToClould()) continue ;
      if(cam_line!="")
      {
        extract_cam_pose(cam_line);
        pcl::transformPointCloud(*scene_cloud, *transformed_scene, cam_T);
      }
      else continue;
      processModels();
    }

    if(cam_line!="")
    {
      extract_cam_pose(cam_line);
      pcl::transformPointCloud(*scene_cloud, *transformed_scene, cam_T);
    }

    if(curr_models.size())
    {
      for(int i=0; i < curr_models.size(); i++)
        *pub_one_pred += curr_models[i];
    }
    if(global_models.size())
    {
      for(int i=0; i < global_models.size(); i++)
      {
        if(confidence_scores[i] > confidence_thresh)
          *pub_multi_pred += global_models[i];
      }
    }

    *pub_one_pred += *scene_cloud;
    pub_one_pred->header.frame_id = "camera_depth_optical_frame";  

    *pub_multi_pred += *transformed_scene;
    pub_multi_pred->header.frame_id = "camera_depth_optical_frame";  

    pcl::toPCLPointCloud2(*pub_one_pred, cloud_filtered);
    pcl_conversions::fromPCL(cloud_filtered, output);
    cloud_pub_one_pred.publish (output);

    pcl::toPCLPointCloud2(*pub_multi_pred, cloud_filtered);
    pcl_conversions::fromPCL(cloud_filtered, output);
    cloud_pub_multi_pred.publish (output);

    ros::spinOnce();
    loop_rate.sleep();
    posefile.close();
  }
  cam_traject_file.close();

  return 0;
}