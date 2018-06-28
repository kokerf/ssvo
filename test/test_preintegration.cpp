#include "global.hpp"
#include "dataset.hpp"
#include "time_tracing.hpp"
#include "optimizer.hpp"
#include "preintegration.hpp"
#include "utils.hpp"
#include "config.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
	//FLAGS_alsologtostderr = true;
	//FLAGS_colorlogtostderr = true;
	//FLAGS_log_prefix = false;
	google::InitGoogleLogging(argv[0]);
	LOG_ASSERT(argc == 2) << "\n Usage : ./test_perintegration dataset_path";

	EuRocDataReader dataset(argv[1]);
	IMUPara::setMeasCov(2.0000e-3, 1.6968e-04);
	IMUPara::setBiasCov(3.0000e-3, 1.9393e-05);
	Vector3d acc_bias(-0.022996, 0.125896, 0.057076);
	Vector3d gyro_bias(-0.002571, 0.021269, 0.076861);
	IMUBias imu_bias_zero(gyro_bias, acc_bias);
	Preintegration init_preint(imu_bias_zero);

	const size_t N = dataset.leftImageSize();
	const double keyframe_duration = 0.30;
	
	size_t imu_idx = 0;
	bool first_time = true;
	EuRocDataReader::Image image_data_last = dataset.leftImage(0);
	Sophus::SE3d Twci;
	for (size_t i = 0; i < N; i++)
	{
		const EuRocDataReader::Image image_data = dataset.leftImage(i);
		if (image_data.timestamp - image_data_last.timestamp < keyframe_duration)
			continue;

		std::cout << std::fixed << std::setprecision(7);
		std::cout << "=== Load Image " << i << ": " << image_data.path << ", time: " << image_data.timestamp << std::endl;
		cv::Mat image = cv::imread(image_data.path, CV_LOAD_IMAGE_UNCHANGED);
		if (image.empty())
			continue;

		EuRocDataReader::GroundTruthData ground_truth;
		bool succeed = dataset.getGroundtruthAtTime(image_data.timestamp, ground_truth);
		if (!succeed) continue;

		std::cout << "Get groundtruth at timestamp: " << ground_truth.timestamp << std::endl;

		Quaterniond quat_wcj(ground_truth.q[0], ground_truth.q[1], ground_truth.q[2], ground_truth.q[3]);
		Vector3d tran_wcj(ground_truth.p[0], ground_truth.p[1], ground_truth.p[2]);
		Sophus::SE3d Twcj(quat_wcj, tran_wcj);

		//! escape imu data at the first time
		if (first_time)
		{
			while(1)
			{
				EuRocDataReader::IMUData data = dataset.imu(imu_idx);
				if (data.timestamp >= image_data.timestamp)
					break;

				imu_idx++;
			}
			image_data_last = image_data;
			Twci = Twcj;
			first_time = false;
			continue;
		}

		std::vector<IMUData> imu_data;
		while (1)
		{
			EuRocDataReader::IMUData dataset_data = dataset.imu(imu_idx);

			if(dataset_data.timestamp >= image_data.timestamp)
				break;

			IMUData data(dataset_data.timestamp,
				Vector3d(dataset_data.gyro[0], dataset_data.gyro[1], dataset_data.gyro[2]),
				Vector3d(dataset_data.acc[0], dataset_data.acc[1], dataset_data.acc[2]));

			imu_data.push_back(data);
			
			imu_idx++;
		}

		Preintegration perint;
		Preintegration::integrate(perint, imu_data, imu_bias_zero, image_data_last.timestamp, image_data.timestamp);

		Sophus::SE3d Tij = Twci.inverse() * Twcj;
		Quaterniond qij = Tij.unit_quaternion();
		std::cout << "Tij[t, q]: " << Tij.translation().transpose() << ", (" << qij.x() << ", " << qij.y() << ", " << qij.z() << ", " << qij.w() << ")" << std::endl;
		std::cout << "Preint:\n" << perint << std::endl;

		if(1)//! test correct
		{
			Preintegration preint_correct = perint;
			Preintegration preint_recalcu;
			IMUBias bias_old = preint_correct.getBias();
			IMUBias bias_new(bias_old.gyro_bias_ + Vector3d(0.01, 0.01, 0.1), bias_old.acc_bias_ + Vector3d(0.1, -0.2, 0.3));

			preint_correct.correct(bias_new);
			Preintegration::integrate(preint_recalcu, imu_data, bias_new, image_data_last.timestamp, image_data.timestamp);
			std::cout << std::fixed << std::setprecision(7);
			std::cout << "Preint by correct:\n" << preint_correct << std::endl;
			std::cout << std::fixed << std::setprecision(7);
			std::cout << "Preint by recalcu:\n" << preint_recalcu << std::endl;
		}

		//! update
		image_data_last = image_data;
		Twci = Twcj;
	}


	return 0;
}