#include "obj_pose_est/kalman_filter.h"

kalman_filter::kalman_filter()
{
    std::cerr << "Start Kalman Filter!!!\n";
}

kalman_filter::~kalman_filter()
{
}


inline Vector6f kalman_filter::subPose(const Vector6f &origin, const Vector6f &pose)
{
    Eigen::Affine3f origin_A = Eigen::Affine3f::Identity();
    origin_A.translation() << origin(0), origin(1), origin(2);
    origin_A.rotate (Eigen::AngleAxisf (origin(3), Eigen::Vector3f::UnitX()));
    origin_A.rotate (Eigen::AngleAxisf (origin(4), Eigen::Vector3f::UnitY()));
    origin_A.rotate (Eigen::AngleAxisf (origin(5), Eigen::Vector3f::UnitZ()));

    Eigen::Affine3f pose_A = Eigen::Affine3f::Identity();
    pose_A.translation() << pose(0), pose(1), pose(2);
    pose_A.rotate (Eigen::AngleAxisf (origin(3), Eigen::Vector3f::UnitX()));
    pose_A.rotate (Eigen::AngleAxisf (origin(4), Eigen::Vector3f::UnitY()));
    pose_A.rotate (Eigen::AngleAxisf (origin(5), Eigen::Vector3f::UnitZ()));

    Eigen::Affine3f result_A;
    result_A.matrix() = origin_A.matrix()*pose_A.matrix().inverse();
    Vector3f rot = result_A.rotation().eulerAngles(0, 1, 2);

    Vector6f result;
    result(0) = result_A.translation()(0); result(1) = result_A.translation()(1); result(2) = result_A.translation()(2);
    result(3) = rot(0); result(4) = rot(1); result(5) = rot(3);
    return result;
}

inline Vector6f kalman_filter::addPose(const Vector6f &origin, const Vector6f &pose)
{
    Eigen::Affine3f origin_A = Eigen::Affine3f::Identity();
    origin_A.translation() << origin(0), origin(1), origin(2);
    origin_A.rotate (Eigen::AngleAxisf (origin(3), Eigen::Vector3f::UnitX()));
    origin_A.rotate (Eigen::AngleAxisf (origin(4), Eigen::Vector3f::UnitY()));
    origin_A.rotate (Eigen::AngleAxisf (origin(5), Eigen::Vector3f::UnitZ()));

    Eigen::Affine3f pose_A = Eigen::Affine3f::Identity();
    pose_A.translation() << pose(0), pose(1), pose(2);
    pose_A.rotate (Eigen::AngleAxisf (origin(3), Eigen::Vector3f::UnitX()));
    pose_A.rotate (Eigen::AngleAxisf (origin(4), Eigen::Vector3f::UnitY()));
    pose_A.rotate (Eigen::AngleAxisf (origin(5), Eigen::Vector3f::UnitZ()));

    Eigen::Affine3f result_A;
    result_A.matrix() = origin_A.matrix()*pose_A.matrix();
    Vector3f rot = result_A.rotation().eulerAngles(0, 1, 2);

    Vector6f result;
    result(0) = result_A.translation()(0); result(1) = result_A.translation()(1); result(2) = result_A.translation()(2);
    result(3) = rot(0); result(4) = rot(1); result(5) = rot(3);
    return result;
}

bool kalman_filter::Kalman_update(const Vector6f &measure_X, const Matrix6f &measure_cov, 
                    Vector6f &est_X, Matrix6f &est_cov)
{
    Matrix6f Rk, Sk, Kk, Ik, Sk_inv;
    Ik.setIdentity();
    Vector6f yk;   

    Vector6f measured, estimated, residual, corrected, correction;

    measured(0) = measure_X(0); measured(1) = measure_X(1); measured(2) = measure_X(2);
    measured(3) = measure_X(3); measured(4) = measure_X(4); measured(5) = measure_X(5);

    estimated(0) = est_X(0); estimated(1) = est_X(1); estimated(2) = est_X(2);
    estimated(3) = est_X(3); estimated(4) = est_X(4); estimated(5) = est_X(5);
 
    bool invertible;
    double det;

    residual = subPose(estimated, measured); 

    Rk = measure_cov;
    Sk = est_cov + Rk;
    FullPivLU<Matrix6f> lu_decomp(Sk);
    invertible = lu_decomp.isInvertible();

    if (!invertible) 
    {
      std::cerr << "Matrix not invertible \n";
    }
    else
    {
      Sk_inv = Sk.inverse();
    }
    yk(0) = residual(0); yk(1) = residual(1); yk(2) = residual(2);
    yk(3) = residual(3); yk(4) = residual(4); yk(5) = residual(5);

    Kk = est_cov*Sk_inv;
    yk = Kk*yk;
    correction(0) = yk(0); correction(1) = yk(1); correction(2) = yk(2);
    correction(3) = yk(3); correction(4) = yk(4); correction(5) = yk(5);  
    corrected = addPose(estimated, correction);
    
    //est_X(0) = corrected(0); est_X(1) = corrected(1); est_X(2) = corrected(2);
    //est_X(3) = corrected(3); est_X(4) = corrected(4); est_X(5) = corrected(5);
    est_X = corrected;
    est_cov = (Ik - Kk)*est_cov;
}
