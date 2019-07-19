#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>

#include "obj_pose_est/model_to_scene.h"
#include "obj_pose_est/ObjectRPE.h"

using namespace cv;
using namespace std;

class rpeCamNode
{
  public:
    rpeCamNode();
    virtual ~rpeCamNode();
    void subcribeTopics();
    void advertiseTopics();
    void depthCallback(const sensor_msgs::Image::ConstPtr& msg);
    void rgbCallback(const sensor_msgs::Image::ConstPtr& msg);
    
    // Process object poses
    bool depthToClould();
    void colorMap(int i, pcl::PointXYZRGB &point);
    void loadModels();
    void getCalibrationParas(std::string dataset);
    void extract_transform_from_quaternion(std::string line, Eigen::Matrix4f &T, int class_index);
    void pose_process();
    
    // Save data
    bool only_save_frames;
    int depth_now, rgb_now;
    int num_frames;
    std::string data_dir, dataset, ObjectRPE_dir;
    std::string depth_topsub, rgb_topsub;
    std::string saved_rgb_dir, saved_depth_dir;

    // Process object poses
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  scene_cloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  pub_cloud;

    std::string depth_path, rgb_path;
    cv::Mat rgb_img, depth_img;
    double fx, fy, cx, cy, depth_factor;

    std::vector<string> object_names; // names of object detected
    std::vector<string> model_paths; // path to model of object detected
    std::vector<string> full_items; // full list of item names in dataset 
    std::vector<Eigen::Matrix4f> transforms;

  private:
   ros::NodeHandle nh_, nh_rgb, nh_depth, nh_cloud, nh_srv;
   ros::Subscriber depth_sub, rgb_sub;
   ros::Publisher cloud_pub;
   ros::ServiceClient client;
   obj_pose_est::ObjectRPE srv;

};

rpeCamNode::rpeCamNode()
{
  nh_ = ros::NodeHandle("~");
  nh_rgb = ros::NodeHandle("~");
  nh_depth = ros::NodeHandle("~");
  nh_cloud = ros::NodeHandle("~");

  nh_depth.getParam("depth_topsub", depth_topsub);
  nh_rgb.getParam("rgb_topsub", rgb_topsub);

  nh_.getParam("only_save_frames", only_save_frames);
  nh_.getParam("num_frames", num_frames);
  nh_.getParam("dataset", dataset);
  nh_.getParam("ObjectRPE_dir", ObjectRPE_dir);
  nh_.getParam("data_dir", data_dir);

  client = nh_srv.serviceClient<obj_pose_est::ObjectRPE>("Seg_Reconst_PoseEst");
  srv.request.ObjectRPE_dir = ObjectRPE_dir;
  srv.request.dataset = dataset;
  srv.request.data_dir = data_dir; 
  srv.request.num_frames = num_frames;

  pub_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
  depth_now = 0; rgb_now = 0;
}

rpeCamNode::~rpeCamNode()
{
};

void rpeCamNode::subcribeTopics()
{
  depth_sub = nh_depth.subscribe (depth_topsub, 1, &rpeCamNode::depthCallback, this);
  rgb_sub = nh_rgb.subscribe (rgb_topsub, 1, &rpeCamNode::rgbCallback, this);  
}

void rpeCamNode::advertiseTopics()
{
  cloud_pub = nh_cloud.advertise<sensor_msgs::PointCloud2> ("myCloud", 1);
}

void rpeCamNode::depthCallback (const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge;

  try
  {
    bridge = cv_bridge::toCvCopy(msg, "32FC1");
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Failed to transform depth image.");
    return;
  }

  if(depth_now < num_frames) 
  {
    cv::Mat depth = bridge->image;
    depth.convertTo(depth, CV_16UC1, 1000.0);
    int now = 1000001 + depth_now;
    saved_depth_dir = data_dir + "/depth/" + std::to_string(now).substr(1, 6) + "-depth.png";
    cv::imwrite( saved_depth_dir, depth );
    depth_now++;
  }

  if(only_save_frames) return;

  if(depth_now==num_frames & rgb_now==num_frames)
  {
    //-----------------------Call service for MaskRCNN and DenseFusion---------------------------
    
    ROS_INFO("ObjectRPE running");
    
    if (client.call(srv))
    {
        ROS_INFO("Result: %ld", (long int)srv.response.ouput);
    }
    else
    {
        ROS_ERROR("Failed to call service ObjectRPE");
    }
    pose_process();
    depth_now=0; rgb_now=0;
  }

  if(pub_cloud->size())
  {
    pcl::PCLPointCloud2 cloud_filtered;
    sensor_msgs::PointCloud2 output;
    pub_cloud->header.frame_id = "camera_depth_optical_frame";  
    pcl::toPCLPointCloud2(*pub_cloud, cloud_filtered);
    pcl_conversions::fromPCL(cloud_filtered, output);
    cloud_pub.publish (output);
  }
}

void rpeCamNode::rgbCallback (const sensor_msgs::Image::ConstPtr& msg)
{
  cv_bridge::CvImageConstPtr bridge;
  try
  {
    bridge = cv_bridge::toCvCopy(msg, "bgr8");    
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Failed to transform rgb image.");
    return;
  }

  if(rgb_now < num_frames) 
  {
      cv::Mat rgb_image;
      rgb_image = bridge->image;
      int now = 1000001 + rgb_now;
      saved_rgb_dir = data_dir + "/rgb/" + std::to_string(now).substr(1, 6) + "-color.png";  
      cv::imwrite( saved_rgb_dir, rgb_image );
      rgb_now++;
      //cv::imshow("RGB image", rgb_image);
      //cv::waitKey(3);
  }
}

bool rpeCamNode::depthToClould()
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

void rpeCamNode::colorMap(int i, pcl::PointXYZRGB &point)
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

void rpeCamNode::loadModels()
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
    *pub_cloud += *color_model_cloud;
  }
}

void rpeCamNode::getCalibrationParas(std::string dataset)
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

void rpeCamNode::extract_transform_from_quaternion(std::string line, Eigen::Matrix4f &T, int class_index)
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

void rpeCamNode::pose_process()
{
  std::string classes_path = data_dir + "/dataset/warehouse/image_sets/classes.txt";
  std::string pose_path = data_dir + "/DenseFusion_Poses.txt";
  std::string model_dir = data_dir + "/dataset/warehouse/models";
  
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

  while (!posefile.eof())
  {
    pub_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    scene_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    model_paths.clear();
    transforms.clear();
    object_names.clear();

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
        std::cerr << line <<  endl;     

        while(true)
        {
          getline (posefile, line);
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
    if(model_paths.size())
    {
      loadModels();
      if(!depthToClould()) continue ;
    }
    *pub_cloud += *scene_cloud;
    std::cerr << "scene: " << scene_cloud->size() << "\n"; 
    std::cerr << "pub: " << pub_cloud->size() << "\n"; 
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "find_transform");
  rpeCamNode mainNode;
  mainNode.subcribeTopics();
  mainNode.advertiseTopics();
  ros::spin();
  return 0;
}