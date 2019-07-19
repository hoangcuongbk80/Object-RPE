#include <ros/package.h>
#include<vector>

// OpenCV specific includes
#include <opencv2/highgui/highgui.hpp>

// PCL specific includes
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/vfh.h>
#include <pcl/features/normal_3d.h>
//plane fitting
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/common/geometry.h>

#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/ply_io.h>


#ifndef MODEL_TO_SCENE_H
#define MODEL_TO_SCENE_H

struct boundingBBox
{
    double length[3];
    pcl::PointXYZ minPoint, maxPoint; //min max Point after trasferring to origin
    pcl::PointXYZ center;
    Eigen::Matrix4f toOrigin; //pcl::transformPointCloud(input, output, toOrigin);
    pcl::PointCloud<pcl::PointXYZ> cornerPoints;

};

struct ObjectModel
{
    std::string name;
	pcl::PointCloud<pcl::PointXYZRGB> points_full;
    pcl::PointCloud<pcl::PointXYZRGB> points_view; 
    boundingBBox OBB;   
};

struct ObjectInstance
{
    std::string name;
	pcl::PointCloud<pcl::PointXYZRGB> points;
    boundingBBox OBB;   
};

class model_to_scene
{
  public:
    model_to_scene();
    virtual ~model_to_scene();

    std::vector<ObjectInstance> instances;
    std::vector<ObjectModel> models;
    
    void computeOBB(const pcl::PointCloud<pcl::PointXYZ> &input, boundingBBox &OBB);
    void processCloud(const std::string cloud_path, 
                      const std::string detected_class_ids_path, 
                      const std::string class_list_path,
                      const std::string model_dir);
    
    void coarseToFineRegistration(const pcl::PointCloud<pcl::PointXYZ> &sourceCloud,
                                  const pcl::PointCloud<pcl::PointXYZ> &modelCloud, 
                                  const pcl::PointCloud<pcl::PointXYZ> &targetCloud,
                                  pcl::PointCloud<pcl::PointXYZ> &registed_source);
    void cloud_to_instances(const pcl::PointCloud<pcl::PointXYZRGB> &input, 
                            std::vector<ObjectInstance> &objects);
    int color_to_instanceID(unsigned char r, unsigned char g, unsigned char b);
    void noiseRemoval(std::vector<ObjectInstance> &objects);
    void load_models(const std::string model_dir,
                     const std::string detected_class_ids_path, 
                     const std::string class_list_path,
                     std::vector<ObjectModel> &models);
    
    double overlapPortion(const pcl::PointCloud<pcl::PointXYZ> &source, 
                          const pcl::PointCloud<pcl::PointXYZ> &target, 
                          const double &max_dist);
};

#endif