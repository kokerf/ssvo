#include "global.hpp"
#include "dataset.hpp"
#include "time_tracing.hpp"
#include "optimizer.hpp"
#include "preintegration.hpp"
#include "utils.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
	FLAGS_alsologtostderr = true;
	FLAGS_colorlogtostderr = true;
	FLAGS_log_prefix = false;
	google::InitGoogleLogging(argv[0]);
	LOG_ASSERT(argc == 2) << "\n Usage : ./test_perintegration dataset_path";

	EuRocDataReader dataset(argv[1]);
	IMUPara::setMeasCov(2.0000e-3, 1.6968e-04);
	IMUPara::setBiasCov(3.0000e-3, 1.9393e-05);

	ssvo::Timer<std::micro> timer;
	const size_t N = dataset.groundtruthSize();
	const double keyframe_duration = 1.0;
	double keyframe_timestamp_last = 0.0;
	size_t imu_idx = 0;
	size_t groundtruth_idx = 0;
	EuRocDataReader::IMUData imu_data_last = dataset.imu(0);
	EuRocDataReader::GroundTruthData ground_truth_last = dataset.groundtruth(0);
	//! align timestamp
	while (ground_truth_last.timestamp < imu_data_last.timestamp)
	{
		groundtruth_idx++;
		ground_truth_last = dataset.groundtruth(groundtruth_idx);
	}
	while (imu_data_last.timestamp < ground_truth_last.timestamp)
	{
		imu_idx++;
		imu_data_last = dataset.imu(imu_idx);
	}

	Quaterniond qwi(ground_truth_last.q[0], ground_truth_last.q[1], ground_truth_last.q[2], ground_truth_last.q[3]);
	Vector3d twi(ground_truth_last.p[0], ground_truth_last.p[1], ground_truth_last.p[2]);
	Sophus::SE3d Twi(qwi, twi);

	Vector3d acc_bias(-0.022996, 0.125896, 0.057076);
	Vector3d gyro_bias(-0.002571, 0.021269, 0.076861);
	IMUBias imu_bias_zero(gyro_bias, acc_bias);
	std::list<Preintegration> preintegration_list;
	for (size_t i = 0; groundtruth_idx < N; i++)
	{
		EuRocDataReader::GroundTruthData ground_truth = dataset.groundtruth(groundtruth_idx);
		while (ground_truth.timestamp - ground_truth_last.timestamp < keyframe_duration &&  groundtruth_idx < N)
		{
			groundtruth_idx++;
			ground_truth = dataset.groundtruth(groundtruth_idx);
		}
		Quaterniond qwj(ground_truth.q[0], ground_truth.q[1], ground_truth.q[2], ground_truth.q[3]);
		Vector3d twj(ground_truth.p[0], ground_truth.p[1], ground_truth.p[2]);
		Sophus::SE3d Twj(qwj, twj);
		Sophus::SE3d Tij = Twi.inverse() * Twj;
		Twi = Twj;

		Quaterniond qij = Tij.unit_quaternion();
		LOG(INFO) << "Start frame " << i << ": , time from " << std::fixed << std::setprecision(7) << ground_truth_last.timestamp  << " to " << ground_truth.timestamp 
				<< "\n pose: " << Tij.translation().transpose() << ", [" << qij.x() << ", " << qij.y() << ", " << qij.z() << ", " << qij.w() << "]" << std::endl;

		Preintegration preintegration(imu_bias_zero);
		bool first_data = true;
		while(1)
		{
			EuRocDataReader::IMUData imu_data = dataset.imu(imu_idx);
			EuRocDataReader::IMUData imu_data_next = dataset.imu(++imu_idx);
			Vector3d acc_meas(imu_data.acc[0], imu_data.acc[1], imu_data.acc[2]);
			Vector3d gyro_meas(imu_data.gyro[0], imu_data.gyro[1], imu_data.gyro[2]);
			double dt = imu_data_next.timestamp - imu_data.timestamp;
			double dt_res = ground_truth.timestamp - imu_data.timestamp;
			if (first_data)
			{
				dt = imu_data_next.timestamp - ground_truth_last.timestamp;
				first_data = false;
			}
			else if (dt_res < dt)
				dt = dt_res;

			preintegration.update(gyro_meas, acc_meas, dt);
			if (imu_data_next.timestamp >= ground_truth.timestamp)
				break;
		}
		LOG(INFO) << preintegration << std::endl;

		ground_truth_last = ground_truth;
	}


	return 0;
}