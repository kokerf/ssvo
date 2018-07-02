#include <glog/logging.h>
#include "optimizer.hpp"
#include "utils.hpp"
#include "preintegration.hpp"

namespace ssvo {

Matrix3d IMUPara::acc_meas_cov_;
Matrix3d IMUPara::gyro_meas_cov_;
Matrix3d IMUPara::acc_bias_rw_cov_;
Matrix3d IMUPara::gyro_bias_rw_cov_;

Preintegration::Preintegration()
{
	reset();
}

Preintegration::Preintegration(const IMUBias &bias) :
	bias_(bias)
{
	reset();
}

void Preintegration::reset()
{
	delta_t_ = 0.0;
	delta_pos_.setZero();
	delta_rot_.setIdentity();
	delta_vel_.setZero();
	jacob_delta_pos_biasacc_.setZero();
	jacob_delta_pos_biasgyro_.setZero();
	jacob_delta_rot_biasgyro_.setZero();
	jacob_delta_vel_biasgyro_.setZero();
	jacob_delta_vel_biasacc_.setZero();
	imu_meas_cov_.setZero();
}

void Preintegration::update(const Vector3d &measured_gyro, const Vector3d &measured_acc, const double dt)
{
	LOG_ASSERT(dt > 1e-9) << "Error dt near zero! dt = " << dt;
	Vector3d gyro = measured_gyro - bias_.gyro_bias_;
	Vector3d acc = measured_acc - bias_.acc_bias_;

	Vector3d dphi = gyro * dt;
	Matrix3d dR = Sophus::SO3d::exp(dphi).matrix();
	Matrix3d dRt = dR.transpose();
	Matrix3d Jr = Sophus::SO3JacobianR(dphi);
	double half_dt2 = 0.5 * dt * dt;

	//! covariance [dP dR dV]
	static const Matrix3d I3x3 = Matrix3d::Identity();
	Matrix3d rot_acc_hat = delta_rot_ * Sophus::SO3d::hat(acc);
	Matrix9d A = Matrix9d::Identity();
	A.block<3, 3>(0, 3) = rot_acc_hat * (-half_dt2);
	A.block<3, 3>(0, 6) = I3x3 * dt;
	A.block<3, 3>(3, 3) = dRt * dt;
	A.block<3, 3>(6, 3) = rot_acc_hat * (-dt);
	Matrix<double, 9, 3> Ba = Matrix<double, 9, 3>::Zero();
	Ba.block<3, 3>(0, 0) = delta_rot_ * half_dt2;
	Ba.block<3, 3>(6, 0) = delta_rot_ * dt;
	Matrix<double, 9, 3> Bg = Matrix<double, 9, 3>::Zero();
	Bg.block<3, 3>(3, 0) = Jr * dt;
	imu_meas_cov_ = A * imu_meas_cov_ * A.transpose();
	imu_meas_cov_.noalias() += Ba * (IMUPara::accMeasCov() / dt) * Ba.transpose();
	imu_meas_cov_.noalias() += Bg * (IMUPara::gyroMeasCov() / dt) * Bg.transpose();

	//! jacobian of bias
	Matrix3d delta_acc_biasgyro = rot_acc_hat * jacob_delta_rot_biasgyro_;
	//! P V R
	jacob_delta_pos_biasacc_.noalias()  += jacob_delta_vel_biasacc_ * dt - delta_rot_ * half_dt2;
	jacob_delta_pos_biasgyro_.noalias() += jacob_delta_vel_biasgyro_ * dt - delta_acc_biasgyro * half_dt2;
	jacob_delta_vel_biasacc_.noalias()  += delta_rot_ * (-dt);
	jacob_delta_vel_biasgyro_.noalias() += delta_acc_biasgyro * (-dt);
	jacob_delta_rot_biasgyro_ = dRt * jacob_delta_rot_biasgyro_ - Jr * dt;

	//! preintegration
	Vector3d roted_acc = delta_rot_ * acc;
	delta_pos_.noalias() += delta_vel_ * dt + roted_acc * half_dt2;
	delta_vel_.noalias() += roted_acc * dt;
	delta_rot_ = delta_rot_ * dR;
	delta_t_ += dt;
}

void Preintegration::correctDeltaBiasGyro(const Vector3d &delta_biasgyro)
{
	bias_.gyro_bias_.noalias() += delta_biasgyro;

	delta_rot_ = delta_rot_ * Sophus::SO3d::exp(jacob_delta_rot_biasgyro_ * delta_biasgyro).matrix();
	delta_vel_.noalias() += jacob_delta_vel_biasgyro_ * delta_biasgyro;
	delta_pos_.noalias() += jacob_delta_pos_biasgyro_ * delta_biasgyro;
}

void Preintegration::correctDeltaBiasAcc(const Vector3d &delta_biasacc)
{
	bias_.acc_bias_.noalias() += delta_biasacc;

	delta_vel_.noalias() += jacob_delta_vel_biasacc_ * delta_biasacc;
	delta_pos_.noalias() += jacob_delta_pos_biasacc_ * delta_biasacc;
}

void Preintegration::correct(const IMUBias &bias)
{
	const Vector3d delta_biasacc = bias.acc_bias_ - bias_.acc_bias_;
	const Vector3d delta_biasgyro = bias.gyro_bias_ - bias_.gyro_bias_;
	bias_ = bias;

	delta_rot_ = delta_rot_ * Sophus::SO3d::exp(jacob_delta_rot_biasgyro_ * delta_biasgyro).matrix();
	delta_vel_.noalias() += jacob_delta_vel_biasacc_ * delta_biasacc + jacob_delta_vel_biasgyro_ * delta_biasgyro;
	delta_pos_.noalias() += jacob_delta_pos_biasacc_ * delta_biasacc + jacob_delta_pos_biasgyro_ * delta_biasgyro;
}

void Preintegration::integrate(Preintegration &preint, const std::vector<IMUData> &imu_data, const IMUBias &bias_last, double timestampi, double timestamej)
{
	LOG_ASSERT(!imu_data.empty()) << "Empty IMU data!!!";

	preint = Preintegration(bias_last);

	//! get time gap between last frame
	double timestamp_last = timestampi;
	{
		const IMUData imu = imu_data.front();
		const double dt = imu.timestamp - timestamp_last;

		LOG_ASSERT(dt >= 0) << "Error timestamp between start time: " << timestamp_last << ", and first imu: " << imu.timestamp;
	}

	//! preintegration
	size_t N = imu_data.size() - 1;
	for (size_t i = 0; i < N; i++)
	{
		const IMUData imu = imu_data[i];
		const double timestamp_cur = imu_data[i + 1].timestamp;
		const double dt = timestamp_cur - timestamp_last;

		LOG_ASSERT(dt >= 0) << "Error timestamp between last imu: " << timestamp_last << ", and current imu: " << timestamp_cur;

		preint.update(imu.gyro, imu.acc, dt);

		timestamp_last = timestamp_cur;
	}

	//! update last data
	{
		const IMUData imu = imu_data.back();
		const double dt = timestamej - timestamp_last;
		LOG_ASSERT(dt >= 0) << "Error timestamp between last imu: " << timestamp_last << ", and end time: " << timestamej;

		preint.update(imu.gyro, imu.acc, dt);
	}
}

std::ostream& operator<<(std::ostream& os, const Preintegration& pint) {
	os << "    deltaTij " << pint.deltaTij() << std::endl;
	Quaterniond qij = Quaterniond(pint.deltaRij());
	os << "    deltaRij [" << qij.x() << "," << qij.y() << "," << qij.z() << "," << qij.w() << "]" << std::endl;
	os << "    deltaPij " << pint.deltaPij().transpose() << std::endl;
	os << "    deltaVij " << pint.deltaVij().transpose() << std::endl;
	os << "    acc bias " << pint.getBias().acc_bias_.transpose() << std::endl;
	os << "    gyrobias " << pint.getBias().gyro_bias_.transpose() << std::endl;
	//os << "    meas cov \n" << std::scientific << pint.getMeasCov() << std::endl;
	return os;
}

}