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



pcl::PointCloud<pcl::PointXYZ>::Ptr     OBBs (new pcl::PointCloud<pcl::PointXYZ>);
std::vector<string>                     OBB_names;
visualization_msgs::MarkerArray         multiMarker;
std::vector<string>                     ObjectNameList;

pcl::PointCloud<pcl::PointXYZRGB>::Ptr  scene_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr  pub_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

std::string depth_path, rgb_path;
cv::Mat rgb_img, depth_img;
double fx, fy, cx, cy, depth_factor;

std::vector<string> object_names; // names of object detected
std::vector<string> model_paths; // path to model of object detected
std::vector<string> full_items; // full list of item names in dataset 
std::vector<Eigen::Matrix4f> transforms;

int OBB_Estimation(pcl::PointCloud<pcl::PointXYZRGB> &cloud)
{
    if(!cloud.size())
    {
      std::cerr << "cloud is empty!" << "\n";
      return 0;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr OBB (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZRGB point;
    
    // Compute principal directions
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(cloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
 
    // Transform the original cloud to the origin where the principal components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPointsProjected (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(cloud, *cloudPointsProjected, projectionTransform);
    // Get the minimum and maximum points of the transformed cloud.
    pcl::PointXYZRGB minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    
    float lengthOBB[3];
    lengthOBB[0] = fabs(maxPoint.x - minPoint.x); //MAX length OBB
    lengthOBB[1] = fabs(maxPoint.y - minPoint.y); //MID length OBB
    lengthOBB[2] = fabs(maxPoint.z - minPoint.z); //MIN length OBB

    pcl::PointCloud<pcl::PointXYZ>::Ptr OBB_Origin(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ OBB_points;
    OBB_points.x = -lengthOBB[0] / 2.0; OBB_points.y = -lengthOBB[1] / 2.0; OBB_points.z = -lengthOBB[2] / 2.0;
    OBB_Origin->push_back(OBB_points); // Min Point
    OBB_points.x = -lengthOBB[0] / 2.0; OBB_points.y = lengthOBB[1] / 2.0; OBB_points.z = -lengthOBB[2] / 2.0;
    OBB_Origin->push_back(OBB_points);
    OBB_points.x = -lengthOBB[0] / 2.0; OBB_points.y = lengthOBB[1] / 2.0; OBB_points.z = lengthOBB[2] / 2.0;
    OBB_Origin->push_back(OBB_points);
    OBB_points.x = -lengthOBB[0] / 2.0; OBB_points.y = -lengthOBB[1] / 2.0; OBB_points.z = lengthOBB[2] / 2.0;
    OBB_Origin->push_back(OBB_points);

    OBB_points.x = lengthOBB[0] / 2.0; OBB_points.y = lengthOBB[1] / 2.0; OBB_points.z = lengthOBB[2] / 2.0;
    OBB_Origin->push_back(OBB_points); //Max point
    OBB_points.x = lengthOBB[0] / 2.0; OBB_points.y = -lengthOBB[1] / 2.0; OBB_points.z = lengthOBB[2] / 2.0;
    OBB_Origin->push_back(OBB_points);
    OBB_points.x = lengthOBB[0] / 2.0; OBB_points.y = -lengthOBB[1] / 2.0; OBB_points.z = -lengthOBB[2] / 2.0;
    OBB_Origin->push_back(OBB_points);
    OBB_points.x = lengthOBB[0] / 2.0; OBB_points.y = lengthOBB[1] / 2.0; OBB_points.z = -lengthOBB[2] / 2.0;
    OBB_Origin->push_back(OBB_points);

    pcl::transformPointCloud(*OBB_Origin, *OBB_Origin, projectionTransform.inverse());

    if(lengthOBB[0] < lengthOBB[1])
    {
        float buf = lengthOBB[0]; lengthOBB[0] = lengthOBB[1]; lengthOBB[1] = buf;
    }
    if(lengthOBB[0] < lengthOBB[2])
    {
        float buf = lengthOBB[0]; lengthOBB[0] = lengthOBB[2]; lengthOBB[2] = buf;
    }
    if(lengthOBB[1] < lengthOBB[2])
    {
        float buf = lengthOBB[1]; lengthOBB[1] = lengthOBB[2]; lengthOBB[2] = buf;
    }
    
    *OBBs += *OBB_Origin;
    OBB_points.x = lengthOBB[0]; OBB_points.y = lengthOBB[1]; OBB_points.z = lengthOBB[2];
    OBBs->push_back(OBB_points);
    std::cerr << "OBB length: " << lengthOBB[0] << " " << lengthOBB[1] << " " << lengthOBB[2] << "\n";
    return 1;
}

void draw_OBBs()
{
    visualization_msgs::Marker OBB;
    geometry_msgs::Point p;

    OBB.header.frame_id = "camera_depth_optical_frame";
    OBB.header.stamp = ros::Time::now();
    OBB.ns = "OBBs";
    OBB.id = 0;
    OBB.type = visualization_msgs::Marker::LINE_LIST;
    OBB.action = visualization_msgs::Marker::ADD;
    OBB.pose.position.x = 0;
    OBB.pose.position.y = 0;
    OBB.pose.position.z = 0;
    OBB.pose.orientation.x = 0.0;
    OBB.pose.orientation.y = 0.0;
    OBB.pose.orientation.z = 0.0;
    OBB.pose.orientation.w = 1.0;
    OBB.scale.x = 0.005; OBB.scale.y = 0.005; OBB.scale.z = 0.005;
    OBB.color.r = 1.0f; OBB.color.g = 1.0f; OBB.color.b = 1.0f; OBB.color.a = 1.0;

    visualization_msgs::Marker ObjectName;
    ObjectName.header.frame_id = "camera_depth_optical_frame";
    ObjectName.header.stamp = ros::Time::now();
    ObjectName.id = 0;
    ObjectName.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    ObjectName.action = visualization_msgs::Marker::ADD;
    ObjectName.pose.orientation.x = 0.0;
    ObjectName.pose.orientation.y = 0.0;
    ObjectName.pose.orientation.z = 0.0;
    ObjectName.pose.orientation.w = 1.0;
    ObjectName.scale.x = 0.05; ObjectName.scale.y = 0.05; ObjectName.scale.z = 0.05;
    ObjectName.color.r = 1.0f; ObjectName.color.g = 1.0f; ObjectName.color.b = 1.0f; ObjectName.color.a = 1.0;

    if(OBBs->size() == 0) 
    { 
      OBB.lifetime = ros::Duration();
      multiMarker.markers.push_back(OBB);
      if(!ObjectNameList.empty())
      {
        for(int i=0; i < ObjectNameList.size(); i++)
        {
          ObjectName.ns  = ObjectNameList[i];
          ObjectName.text = "";
          ObjectName.lifetime = ros::Duration();
          multiMarker.markers.push_back(ObjectName);
        }
        ObjectNameList.clear();
      }      
      return;
    };

    if(!ObjectNameList.empty())
      {
        for(int i=0; i < ObjectNameList.size(); i++)
        {
          ObjectName.ns  = ObjectNameList[i];
          ObjectName.text = "";
          ObjectName.lifetime = ros::Duration();
          multiMarker.markers.push_back(ObjectName);
        }
      }
    ObjectNameList.clear();
 
    for(int k = 0; k < OBB_names.size(); k++)
    {
       int begin = k*9; int stop = begin + 8;
       for(int i = begin; i < stop; i++)
       {
          if(i == begin + 3 || i == begin + 7)
          {
             p.x = OBBs->points[i].x;
             p.y = OBBs->points[i].y; p.z = OBBs->points[i].z;
             OBB.points.push_back(p);
             p.x = OBBs->points[i-3].x;
             p.y = OBBs->points[i-3].y; p.z = OBBs->points[i-3].z;
             OBB.points.push_back(p);
             if(i == begin + 3)
             {
                p.x = OBBs->points[i].x;
                p.y = OBBs->points[i].y; p.z = OBBs->points[i].z;
                OBB.points.push_back(p);
                p.x = OBBs->points[begin + 5].x;
                p.y = OBBs->points[begin + 5].y; 
                p.z = OBBs->points[begin + 5].z;
                OBB.points.push_back(p);
             }
             if(i == begin + 7)
             {
                p.x = OBBs->points[i].x;
                p.y = OBBs->points[i].y; p.z = OBBs->points[i].z;
                OBB.points.push_back(p);
                p.x = OBBs->points[begin + 1].x;
                p.y = OBBs->points[begin + 1].y; 
                p.z = OBBs->points[begin + 1].z;
                OBB.points.push_back(p);
             }
          }
          else
          {
             p.x = OBBs->points[i].x;
             p.y = OBBs->points[i].y; p.z = OBBs->points[i].z;
             OBB.points.push_back(p);
             p.x = OBBs->points[i+1].x;
             p.y = OBBs->points[i+1].y; p.z = OBBs->points[i+1].z;
             OBB.points.push_back(p);
             if(i == begin + 0)
             {
                p.x = OBBs->points[i].x;
                p.y = OBBs->points[i].y; p.z = OBBs->points[i].z;
                OBB.points.push_back(p);
                p.x = OBBs->points[begin + 6].x;
                p.y = OBBs->points[begin + 6].y; 
                p.z = OBBs->points[begin + 6].z;
                OBB.points.push_back(p);
             }
             if(i == begin + 2)
             {
                p.x = OBBs->points[i].x;
                p.y = OBBs->points[i].y; p.z = OBBs->points[i].z;
                OBB.points.push_back(p);
                p.x = OBBs->points[begin + 4].x;
                p.y = OBBs->points[begin + 4].y; 
                p.z = OBBs->points[begin + 4].z;
                OBB.points.push_back(p);
             }
          }
       }
       
       ostringstream convert;
       convert << k;
       ObjectName.ns = OBB_names[k] + convert.str();
       ObjectName.pose.position.x = OBBs->points[begin + k].x;
       ObjectName.pose.position.y = OBBs->points[begin + k].y;
       ObjectName.pose.position.z = OBBs->points[begin + k].z;
       ObjectName.color.a = 1.0;
       ObjectName.text = OBB_names[k];
       ObjectNameList.push_back(ObjectName.text);

       ObjectName.lifetime = ros::Duration();
       multiMarker.markers.push_back(ObjectName);
    }

    OBB.lifetime = ros::Duration();
    multiMarker.markers.push_back(OBB);

}

void depthToClould()
{
  depth_img = cv::imread(depth_path, -1);
  rgb_img = cv::imread(rgb_path, -1);

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

void loadModels()
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr  model_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr  color_model_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  for(int i=0; i < model_paths.size(); i++)
  {
    pcl::io::loadPLYFile<pcl::PointXYZ> (model_paths[i], *model_cloud);
    pcl::transformPointCloud(*model_cloud, *model_cloud, transforms[i]);
    copyPointCloud(*model_cloud, *color_model_cloud);
    for(int k=0; k < color_model_cloud->size(); k++)
    {
      colorMap(i+1, color_model_cloud->points[k]);
    }
    OBB_Estimation(*color_model_cloud);
    OBB_names.push_back(object_names[i]);
    *pub_cloud += *color_model_cloud;
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


int main(int argc, char** argv)
{
  ros::init(argc, argv, "ObjectRPE");

  ros::NodeHandle n_, nh_, cloud_n, obb_n;
  ros::Publisher cloud_pub = cloud_n.advertise<sensor_msgs::PointCloud2> ("myCloud", 1);
  ros::Publisher OBBs_pub = obb_n.advertise<visualization_msgs::MarkerArray>( "OBBs", 1);
  ros::Rate loop_rate(10);

  std::string dataset, ObjectRPE_dir, data_dir; 
  int num_frames;

  n_ = ros::NodeHandle("~");
  n_.getParam("ObjectRPE_dir", ObjectRPE_dir);
  n_.getParam("dataset", dataset);
  n_.getParam("data_dir", data_dir);
  n_.getParam("num_frames", num_frames);  

  getCalibrationParas(dataset);

  //-----------------------Call service for MaskRCNN and DenseFusion---------------------------

  ros::ServiceClient client = nh_.serviceClient<obj_pose_est::ObjectRPE>("Seg_Reconst_PoseEst");
  obj_pose_est::ObjectRPE srv;

  srv.request.ObjectRPE_dir = ObjectRPE_dir;
  srv.request.dataset = dataset;
  srv.request.data_dir = data_dir;
  srv.request.num_frames = num_frames;

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
 
  //-----------------------------------Process predictions-------------------------------------
  
  std::string classes_path = data_dir + "/dataset/warehouse/image_sets/classes.txt";
  std::string pose_path = data_dir + "/DenseFusion_Poses.txt";
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
  
  ifstream posefile (pose_path);    
  string line;
  bool firstLine = true;                             

  while (ros::ok())
  {
    pub_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    OBBs.reset(new pcl::PointCloud<pcl::PointXYZ>);
    model_paths.clear();
    transforms.clear();
    OBB_names.clear();

    if (posefile.is_open())                     
    {
      if (!posefile.eof())                 
      {
        if(firstLine) 
        {
          getline (posefile, line);
          firstLine = false;
        }
        depth_path = data_dir + "/depth/" + line + "-depth.png";
        rgb_path = data_dir + "/rgb/" + line + "-color.png";            
        std::cerr << rgb_path << "\n";
        while(true)
        {
          getline (posefile, line);
          std::cerr << line <<  endl;          
          if(line.length() == 6 | line=="") break;
          int cls_index = std::stoi(line) - 1;
          std::string model_path = model_dir + "/" + full_items[cls_index] + "/points.ply";
          object_names.push_back(full_items[cls_index]);
          model_paths.push_back(model_path);
          getline (posefile, line);
          Eigen::Matrix4f T(Eigen::Matrix4f::Identity());
          extract_transform_from_quaternion(line, T, cls_index);          
          transforms.push_back(T);
        }
      }
    }
    else 
    {
      std::cerr << "Unable to open file";
      exit(0);
    }
  
    loadModels();
    depthToClould();
    draw_OBBs();

    *pub_cloud += *scene_cloud;
    pub_cloud->header.frame_id = "camera_depth_optical_frame";  

    pcl::toPCLPointCloud2(*pub_cloud, cloud_filtered);
    pcl_conversions::fromPCL(cloud_filtered, output);
    cloud_pub.publish (output);

    OBBs_pub.publish(multiMarker);
    
    ros::spinOnce();
    loop_rate.sleep();
  }

  posefile.close();
  return 0;
}