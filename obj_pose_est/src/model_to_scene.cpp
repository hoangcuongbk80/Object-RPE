#include "obj_pose_est/model_to_scene.h"

model_to_scene::model_to_scene()
{
    std::cerr << "Start model to scene!!!\n";
}

model_to_scene::~model_to_scene()
{
}

int model_to_scene::color_to_instanceID(unsigned char r, unsigned char g, unsigned char b)
{
    if(r==128 & g==128 & b==0)   return 1;
    else if(r==0   & g==128 & b==128) return 2;
    else if(r==128 & g==0   & b==128) return 3;
    else if(r==128 & g==0   & b==0)   return 4;
    else if(r==0   & g==128 & b==0)   return 5;
    else if(r==0   & g==0   & b==128) return 6;
    else if(r==255 & g==255 & b==0)   return 7;
    else if(r==255 & g==0   & b==255) return 8;
    else if(r==0   & g==255 & b==255) return 9;
    else if(r==255 & g==0   & b==0)   return 10;
    else if(r==0   & g==255 & b==0)   return 11;
    else if(r==0   & g==0   & b==255) return 12;
    else if(r==92  & g==112 & b==92)  return 13;
    else if(r==0   & g==0   & b==70)  return 14;
    else if(r==0   & g==60  & b==100) return 15;
    else if(r==0   & g==80  & b==100) return 16;
    else if(r==0   & g==0   & b==230) return 17;
    else if(r==119 & g==11  & b==32)  return 18;
    else if(r==0   & g==0   & b==121) return 19;    
    else return 0;
}

void model_to_scene::cloud_to_instances(const pcl::PointCloud<pcl::PointXYZRGB> &input, std::vector<ObjectInstance> &objects)
{
    for(int i=0; i < input.size(); i++)
    {
        pcl::PointXYZRGB point;
        point = input.points[i]; 
        int objID;

        if(point.r==192 & point.g==192 & point.b==192) continue;
        else objID = color_to_instanceID(point.r, point.g, point.b);
        if(!objID) continue;

        pcl::PointXYZ pointXYZ;
        pointXYZ.x=point.x; pointXYZ.y=point.y; pointXYZ.z=point.z;
        objects[objID-1].points.push_back(point);
    }
}

void model_to_scene::noiseRemoval(std::vector<ObjectInstance> &objects)
{
    for(int i=0 ; i < objects.size(); i++)
    {
        if(objects[i].points.size() == 0) continue;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>());
        pcl::copyPointCloud(objects[i].points, *cloud);
        
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
        std::vector<pcl::PointIndices> cluster_indices;
        tree->setInputCloud(cloud);
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(0.02);
        ec.setMinClusterSize(500);
        ec.setMaxClusterSize(9999999);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        objects[i].points.clear();
        for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	    {
            for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
            {
                objects[i].points.push_back(cloud->points[*pit]);
            }
        }

    }
}

void model_to_scene::load_models(const std::string model_dir,
                 const std::string detected_class_ids_dir, 
                 const std::string class_list_dir,
                 std::vector<ObjectModel> &models)
{
    std::vector<std::string> object_name_list;
    std::ifstream class_file(class_list_dir);
    std::string str; 
    while (std::getline(class_file, str))
    {
        object_name_list.push_back(str);
    }

    std::vector<int> detected_object_ids;
    std::ifstream file(detected_class_ids_dir);
    while (std::getline(file, str))
    {
        detected_object_ids.push_back(stoi(str));
    }

    ObjectModel model;
    if(!models.empty()) models.clear();
    for(int i=0 ; i < detected_object_ids.size(); i++) models.push_back(model);

    for(int i=0; i < models.size(); i++)
    {
        int class_id = detected_object_ids[i];
        std::string model_path = model_dir + object_name_list[class_id-1] + "/points_view1.ply";
        pcl::io::loadPLYFile<pcl::PointXYZRGB> (model_path, models[i].points_view);
        model_path = model_dir + object_name_list[class_id-1] + "/points.ply";
        pcl::io::loadPLYFile<pcl::PointXYZRGB> (model_path, models[i].points_full);
    }
}

void model_to_scene::processCloud(const std::string cloud_path, 
                                  const std::string detected_class_ids_path, 
                                  const std::string class_list_path,
                                  const std::string model_dir)
{
    load_models(model_dir, detected_class_ids_path, class_list_path, models);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::io::loadPLYFile<pcl::PointXYZRGB> (cloud_path, *cloud);

    ObjectInstance instance;
    if(!instances.empty()) instances.clear();
    for(int i=0 ; i < models.size(); i++) instances.push_back(instance);
    cloud_to_instances(*cloud, instances);
    noiseRemoval(instances);

    for (int i=0; i < instances.size(); i++)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr source (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr model (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr registed_source (new pcl::PointCloud<pcl::PointXYZ>);
        
        pcl::copyPointCloud(instances[i].points, *target);
        pcl::copyPointCloud(models[i].points_view, *source);
        pcl::copyPointCloud(models[i].points_full, *model);
        coarseToFineRegistration(*source, *model, *target, *registed_source);
        pcl::copyPointCloud(*registed_source, models[i].points_full);
        std::cerr << "Instance " << i << "\n"  << "Registed!\n";
    }
    std::cerr << "Object Poses Estimation Done!\n";
}

void model_to_scene::computeOBB(const pcl::PointCloud<pcl::PointXYZ> &input, boundingBBox &OBB)
{
    // Compute principal directions
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(input, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(input, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

    // Transform the original cloud to the origin where the principal components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());

    // Transform the original cloud to the origin where the principal components correspond to the axes.
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::transformPointCloud(input, *cloudPointsProjected, projectionTransform);

    // Get the minimum and maximum points of the transformed cloud.
    pcl::PointXYZ minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    
    OBB.length[0] = maxPoint.x - minPoint.x; //MAX length OBB
    OBB.length[1] = maxPoint.y - minPoint.y; //MID length OBB
    OBB.length[2] = maxPoint.z - minPoint.z; //MIN length OBB

    if(OBB.length[0] < OBB.length[1])
    {
        float buf = OBB.length[0]; OBB.length[0] = OBB.length[1]; 
        OBB.length[1] = buf;

        Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
        transform_2.rotate (Eigen::AngleAxisf (M_PI/2.0, Eigen::Vector3f::UnitZ()));
        pcl::transformPointCloud (*cloudPointsProjected, *cloudPointsProjected, transform_2);
        projectionTransform = transform_2.matrix()*projectionTransform;
    }
    if(OBB.length[0] < OBB.length[2])
    {
        float buf = OBB.length[0]; OBB.length[0] = OBB.length[2]; 
        OBB.length[2] = buf;

        Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
        transform_2.rotate (Eigen::AngleAxisf (M_PI/2.0, Eigen::Vector3f::UnitY()));
        pcl::transformPointCloud (*cloudPointsProjected, *cloudPointsProjected, transform_2);
        projectionTransform = transform_2.matrix()*projectionTransform;
    }
    if(OBB.length[1] < OBB.length[2])
    {
        float buf = OBB.length[1]; OBB.length[1] = OBB.length[2]; 
        OBB.length[2] = buf;

        Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();
        transform_2.rotate (Eigen::AngleAxisf (M_PI/2.0, Eigen::Vector3f::UnitX()));
        pcl::transformPointCloud (*cloudPointsProjected, *cloudPointsProjected, transform_2);
        projectionTransform = transform_2.matrix()*projectionTransform;
    }

    pcl::getMinMax3D(*cloudPointsProjected, OBB.minPoint, OBB.maxPoint);
    OBB.toOrigin = projectionTransform;

    pcl::PointXYZ OBB_points;

    OBB.cornerPoints.push_back(OBB.minPoint); // Min Point
    OBB_points.x = OBB.minPoint.x; OBB_points.y = OBB.maxPoint.y; OBB_points.z = OBB.minPoint.z;
    OBB.cornerPoints.push_back(OBB_points);
    OBB_points.x = OBB.minPoint.x; OBB_points.y = OBB.maxPoint.y; OBB_points.z = OBB.maxPoint.z;
    OBB.cornerPoints.push_back(OBB_points);
    OBB_points.x = OBB.minPoint.x; OBB_points.y = OBB.minPoint.y; OBB_points.z = OBB.maxPoint.z;
    OBB.cornerPoints.push_back(OBB_points);

    OBB.cornerPoints.push_back(OBB.maxPoint); //Max point
    OBB_points.x = OBB.maxPoint.x; OBB_points.y = OBB.minPoint.y; OBB_points.z = OBB.maxPoint.z;
    OBB.cornerPoints.push_back(OBB_points);
    OBB_points.x = OBB.maxPoint.x; OBB_points.y = OBB.minPoint.y; OBB_points.z = OBB.minPoint.z;
    OBB.cornerPoints.push_back(OBB_points);
    OBB_points.x = OBB.maxPoint.x; OBB_points.y = OBB.maxPoint.y; OBB_points.z = OBB.minPoint.z;
    OBB.cornerPoints.push_back(OBB_points);

    pcl::transformPointCloud(OBB.cornerPoints, OBB.cornerPoints, projectionTransform.inverse());

    OBB.center.x = 0; OBB.center.y = 0; OBB.center.z = 0;
    for (int i = 0; i < OBB.cornerPoints.size(); i++)
    {
        OBB.center.x += OBB.cornerPoints[i].x;
        OBB.center.y += OBB.cornerPoints[i].y;
        OBB.center.z += OBB.cornerPoints[i].z;
    }
    OBB.center.x = OBB.center.x / OBB.cornerPoints.size();
    OBB.center.y = OBB.center.y / OBB.cornerPoints.size();
    OBB.center.z = OBB.center.z / OBB.cornerPoints.size();
}


double model_to_scene::overlapPortion(const pcl::PointCloud<pcl::PointXYZ> &source, 
                                          const pcl::PointCloud<pcl::PointXYZ> &target, 
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

void model_to_scene::coarseToFineRegistration(const pcl::PointCloud<pcl::PointXYZ> &sourceCloud,
                                              const pcl::PointCloud<pcl::PointXYZ> &modelCloud,
                                              const pcl::PointCloud<pcl::PointXYZ> &targetCloud,
                                              pcl::PointCloud<pcl::PointXYZ> &registed_source)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr source (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr model (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>);

    //pcl::copyPointCloud(model, *targetCloud);

    boundingBBox source_OBB, target_OBB;
    computeOBB(sourceCloud, source_OBB);
    computeOBB(targetCloud, target_OBB);
    pcl::transformPointCloud(sourceCloud, *source, source_OBB.toOrigin);
    pcl::transformPointCloud(modelCloud, *model, source_OBB.toOrigin);
    pcl::transformPointCloud(targetCloud, *target, target_OBB.toOrigin);

    
    double bestScore = -9999999;
    Eigen::Matrix4f bestCoarseMat (Eigen::Matrix4f::Identity());;
    Eigen::Matrix4f bestFineMat (Eigen::Matrix4f::Identity());;

    for(double RX = 0; RX <= M_PI; RX+=M_PI/2)
    for(double RY = 0; RY <= M_PI; RY+=M_PI/2)
    for(double RZ = 0; RZ <= M_PI; RZ+=M_PI)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr rot_source (new pcl::PointCloud<pcl::PointXYZ>);
        
        Eigen::Affine3f ROT = Eigen::Affine3f::Identity();
        ROT.rotate (Eigen::AngleAxisf (RX, Eigen::Vector3f::UnitX()));
        ROT.rotate (Eigen::AngleAxisf (RY, Eigen::Vector3f::UnitY()));
        ROT.rotate (Eigen::AngleAxisf (RZ, Eigen::Vector3f::UnitZ()));                        
        pcl::transformPointCloud (*source, *rot_source, ROT);
    
        pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
        icp.setInputSource(rot_source);
        icp.setInputTarget(target);
        icp.setMaximumIterations (100);
        icp.setMaxCorrespondenceDistance(0.2);
        icp.setRANSACOutlierRejectionThreshold(1);
        icp.align(registed_source);

        double overlap_dst_thresh = 0.03;
        double overlapScore = overlapPortion(*target, registed_source, overlap_dst_thresh);
        std::cerr << "overlap score: " << overlapScore << "\n";
        //if(overlapScore < overlap_score_thresh) continue; 

        if(bestScore < overlapScore)                                             
        {
            bestCoarseMat = ROT.matrix();
            bestFineMat = icp.getFinalTransformation();
            bestScore = overlapScore;
        }
    }
    Eigen::Matrix4f finalMat = bestFineMat*bestCoarseMat*target_OBB.toOrigin;
    pcl::transformPointCloud(*model, registed_source, finalMat.inverse());
}