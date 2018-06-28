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
	LOG_ASSERT(argc == 3) << "\n Usage : ./test_perintegration config_file dataset_path";

	ssvo::Config::file_name_ = argv[1];

	EuRocDataReader dataset(argv[2]);
	IMUPara::setMeasCov(2.0000e-3, 1.6968e-04);
	IMUPara::setBiasCov(3.0000e-3, 1.9393e-05);
	Vector3d acc_bias(-0.022996, 0.125896, 0.057076);
	Vector3d gyro_bias(-0.002571, 0.021269, 0.076861);
	IMUBias imu_bias_zero(gyro_bias, acc_bias);
	Preintegration init_preint(imu_bias_zero);

	ssvo::Timer<std::micro> timer;
	const size_t N = dataset.leftImageSize();
	const double keyframe_duration = 0.30;


	std::vector<Frame::Ptr> all_frames;

	size_t imu_idx = 0;
	for (size_t i = 0; i < N; i++)
	{
		const EuRocDataReader::Image image_data = dataset.leftImage(i);
		std::cout << std::fixed << std::setprecision(7);
		std::cout << "=== Load Image " << i << ": " << image_data.path << ", time: " << image_data.timestamp << std::endl;
		cv::Mat image = cv::imread(image_data.path, CV_LOAD_IMAGE_UNCHANGED);
		if (image.empty())
			continue;

		EuRocDataReader::GroundTruthData ground_truth;
		bool succeed = dataset.getGroundtruthAtTime(image_data.timestamp, ground_truth);
		if (!succeed) continue;

		std::cout << "Get groundtruth at timestamp: " << ground_truth.timestamp << std::endl;

		Quaterniond quat_wc(ground_truth.q[0], ground_truth.q[1], ground_truth.q[2], ground_truth.q[3]);
		Vector3d tran_wc(ground_truth.p[0], ground_truth.p[1], ground_truth.p[2]);
		Sophus::SE3d Twc(quat_wc, tran_wc);

		Frame::Ptr frame_cur = Frame::create(image, image_data.timestamp, nullptr);
		frame_cur->setPose(Twc);

		//! set first frame
		if (all_frames.empty())
		{
			while (1)
			{
				EuRocDataReader::IMUData data = dataset.imu(imu_idx);
				if (data.timestamp >= frame_cur->timestamp_)
					break;

				imu_idx++;
			}

			frame_cur->setPreintergration(init_preint);
			all_frames.push_back(frame_cur);
			continue;
		}

		std::vector<IMUData> imu_datas;
		while (1)
		{
			EuRocDataReader::IMUData data = dataset.imu(imu_idx);

			if (data.timestamp >= frame_cur->timestamp_)
				break;

			IMUData imu_data(data.timestamp, Vector3d(data.gyro[0], data.gyro[1], data.gyro[2]), Vector3d(data.acc[0], data.acc[1], data.acc[2]));
			imu_datas.push_back(imu_data);

			imu_idx++;
		}

		frame_cur->setIMUData(imu_datas);
		frame_cur->computeIMUPreintegrationSinceLastFrame(all_frames.back());

		Sophus::SE3d Tij = all_frames.back()->Tcw() * frame_cur->Twc();
		Quaterniond qij = Tij.unit_quaternion();
		std::cout << "Tij[t, q]: " << Tij.translation().transpose() << ", (" << qij.x() << ", " << qij.y() << ", " << qij.z() << ", " << qij.w() << ")" << std::endl;
		std::cout << "Preint:\n" << frame_cur->getPreintergration() << std::endl;

		// {//! test correct
		// 	Preintegration preint_old = frame_cur->getPreintergration();
		// 	Preintegration preint_new;
		// 	IMUBias bias_old = preint_old.getBias();
		// 	IMUBias bias_new(bias_old.gyro_bias_ + Vector3d(0.01, 0.01, 0.1), bias_old.acc_bias_ + Vector3d(0.1, -0.2, 0.3));

		// 	preint_old.correct(bias_new);
		// 	Preintegration::integrate(preint_new, frame_cur->getIMUData(), bias_new, all_frames.back()->timestamp_, frame_cur->timestamp_);
		// 	std::cout << std::fixed << std::setprecision(7);
		// 	std::cout << "Preint by correct :\n" << preint_old << std::endl;
		// 	std::cout << std::fixed << std::setprecision(7);
		// 	std::cout << "Preint by repreint:\n" << preint_new << std::endl;
		// }

		all_frames.push_back(frame_cur);
		Optimizer::sloveInitialGyroBias(all_frames);
	}


	return 0;
}