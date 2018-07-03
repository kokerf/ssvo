#ifndef _IMU_DATA_HPP_
#define _IMU_DATA_HPP_

#include <Eigen/Core>

namespace ssvo {

using namespace Eigen;

class IMUPara {
public:

	#define g0 9.78046
	
	static void setMeasCov(double acc_noise_sigma, double gyro_noise_sigma)
	{
		init_ |= 0x01;
		acc_meas_cov_ = Matrix3d::Identity() * (acc_noise_sigma * acc_noise_sigma);
		gyro_meas_cov_ = Matrix3d::Identity() * (gyro_noise_sigma * gyro_noise_sigma);
	}
	
	static void setBiasCov(double acc_bias_rw_sigma, double gyro_bias_rw_sigma) 
	{
		init_ |= 0x02;
		acc_bias_rw_cov_ = Matrix3d::Identity() * (acc_bias_rw_sigma * acc_bias_rw_sigma);
		gyro_bias_rw_cov_ = Matrix3d::Identity() * (gyro_bias_rw_sigma * gyro_bias_rw_sigma);
	}

	static void setGravity(double gravity)
	{
		init_ |= 0x04;
		gravity_ = gravity;
	}

	static void setGravity(double latitude, double altitude)
	{
		init_ |= 0x04;
		gravity_ = g0 * (1 + 5.3024e-3 * std::sin(latitude)*std::sin(latitude) - 5.9e-6 * std::sin(2 * latitude) * std::sin(2 * latitude)) - 3.86e-6*altitude;
	}

	static const Matrix3d & accMeasCov() { assert((init_ & 0x01) == 0x01); return acc_meas_cov_; }
	static const Matrix3d & gyroMeasCov() { assert((init_ & 0x01) == 0x01); return gyro_meas_cov_; }
	static const Matrix3d & accBiasRWCov() { assert((init_ & 0x02) == 0x02); return acc_bias_rw_cov_; }
	static const Matrix3d & gyroBiasRWCov() { assert((init_ & 0x02) == 0x02); return gyro_bias_rw_cov_; }
	static const double gravity() { assert((init_ & 0x04) == 0x04); return gravity_; }

private:

	static unsigned int init_;

	static Matrix3d acc_meas_cov_;
	static Matrix3d gyro_meas_cov_;
	static Matrix3d acc_bias_rw_cov_;	//! random walk
	static Matrix3d gyro_bias_rw_cov_;

	static double gravity_;
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
