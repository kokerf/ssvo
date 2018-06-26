#ifndef _IMU_DATA_HPP_
#define _IMU_DATA_HPP_

#include <Eigen/Core>

namespace ssvo {

using namespace Eigen;

class IMUPara {
public:
	
	static void setMeasCov(double acc_noise_sigma, double gyro_noise_sigma)
	{
		acc_meas_cov_ = Matrix3d::Identity() * (acc_noise_sigma * acc_noise_sigma);
		gyro_meas_cov_ = Matrix3d::Identity() * (gyro_noise_sigma * gyro_noise_sigma);
	}
	
	static void setBiasCov(double acc_bias_rw_sigma, double gyro_bias_rw_sigma) 
	{
		acc_bias_rw_cov_ = Matrix3d::Identity() * (acc_bias_rw_sigma * acc_bias_rw_sigma);
		gyro_bias_rw_cov_ = Matrix3d::Identity() * (gyro_bias_rw_sigma * gyro_bias_rw_sigma);
	}

	static const Matrix3d & accMeasCov() { return acc_meas_cov_; }
	static const Matrix3d & gyroMeasCov() { return gyro_meas_cov_; }
	static const Matrix3d & accBiasRWCov() { return acc_bias_rw_cov_; }
	static const Matrix3d & gyroBiasRWCov() { return gyro_bias_rw_cov_; }

private:

	static Matrix3d acc_meas_cov_;
	static Matrix3d gyro_meas_cov_;
	static Matrix3d acc_bias_rw_cov_;	//! random walk
	static Matrix3d gyro_bias_rw_cov_;

};

struct IMUData {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	double timestamp;
	Vector3d gyro;
	Vector3d acc;

	IMUData(double _timestamp, const Vector3d &_gyro, const Vector3d &_acc) :
		timestamp(_timestamp), gyro(_gyro), acc(_acc) {}
};

struct IMUBias {
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Vector3d gyro_bias_;
	Vector3d acc_bias_;

	IMUBias() :
		gyro_bias_(0.0, 0.0, 0.0), acc_bias_(0.0, 0.0, 0.0) {}

	IMUBias(const Vector3d &gyro_bias, const Vector3d &acc_bias) : 
		gyro_bias_(gyro_bias), acc_bias_(acc_bias) {}
};

}

#endif //_IMU_DATA_HPP_
