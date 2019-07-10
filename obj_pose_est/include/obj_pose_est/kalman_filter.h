#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cmath>

// PCL
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl/io/ply_io.h>

using namespace std;
using namespace Eigen;


typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 6, 6> Matrix6f;

#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

class kalman_filter
{
  public:
    kalman_filter();
    virtual ~kalman_filter();

    bool Kalman_update(const Vector6f &measure_X, const Matrix6f &measure_cov, Vector6f &est_X, Matrix6f &est_cov);
    inline Vector6f addPose(const Vector6f &origin, const Vector6f &pose);
    inline Vector6f subPose(const Vector6f &origin, const Vector6f &pose);
};

#endif