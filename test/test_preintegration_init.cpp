#include <fstream>
#include "global.hpp"
#include "dataset.hpp"
#include "time_tracing.hpp"
#include "optimizer.hpp"
#include "preintegration.hpp"
#include "utils.hpp"
#include "config.hpp"

using namespace ssvo;

class IMUInitializer{
public:

	IMUInitializer(const EuRocDataReader *dataset, AbstractCamera::Ptr &camera) :
		dataset_(dataset), camera_(camera), imu_idx_(0) {}

	bool getTruthPose(const double timestamp, double &gt_timestamp, Sophus::SE3d & Twc)
	{
		EuRocDataReader::GroundTruthData ground_truth;
		bool succeed = dataset_->getGroundtruthAtTime(timestamp, ground_truth);
		if (!succeed) return false;

		Quaterniond quat_wc(ground_truth.q[0], ground_truth.q[1], ground_truth.q[2], ground_truth.q[3]);
		Vector3d tran_wc(ground_truth.p[0], ground_truth.p[1], ground_truth.p[2]);
		Twc = Sophus::SE3d(quat_wc, tran_wc);
        gt_timestamp = ground_truth.timestamp;
		return true;
	}

	void addNewFrame(const cv::Mat &image, const Sophus::SE3d &Twc, double timestamp, double random = 0.0, bool verbose = false)
	{
		Sophus::SE3d Twc_cur = Twc;
        if (random != 0.0)
        {
			SE3d Twc_last;
			if(!all_frames_.empty()) Twc_last = all_frames_.back()->Twc();
            SE3d Trc = Twc_last.inverse() * Twc_cur;
            SE3d::Tangent pertur = SE3d::Tangent::Random() * random;
            Trc = Trc * SE3d::exp(pertur);
            Twc_cur = Twc_last * Trc;
        }

		Frame::Ptr frame_cur = Frame::create(image, timestamp, camera_);
		frame_cur->setPose(Twc_cur);

		//! set first frame
		if (all_frames_.empty())
		{
			while (1)
			{
				EuRocDataReader::IMUData data = dataset_->imu(imu_idx_);
				if (data.timestamp >= frame_cur->timestamp_) break;
				imu_idx_++;
			}

            Preintegration init_preint;
			frame_cur->setPreintergration(init_preint);
			all_frames_.push_back(frame_cur);
			return;
		}

		std::vector<IMUData> imu_datas;
		while (1)
		{
			EuRocDataReader::IMUData data = dataset_->imu(imu_idx_);

			if (data.timestamp >= frame_cur->timestamp_) break;

			IMUData imu_data(data.timestamp, Vector3d(data.gyro[0], data.gyro[1], data.gyro[2]), Vector3d(data.acc[0], data.acc[1], data.acc[2]));
			imu_datas.push_back(imu_data);

			imu_idx_++;
		}

		frame_cur->setIMUData(imu_datas);
		frame_cur->computeIMUPreintegrationSinceLastFrame(all_frames_.back());

		if(verbose)
		{
            Sophus::SE3d Tcb = Sophus::SE3d(camera_->T_CB());
			Sophus::SE3d Tij = Tcb.inverse() * all_frames_.back()->Tcw() * frame_cur->Twc() * Tcb;
			Quaterniond qij = Tij.unit_quaternion();
			std::cout << "Tij[t, q]: " << Tij.translation().transpose() << ", (" << qij.x() << ", " << qij.y() << ", " << qij.z() << ", " << qij.w() << ")" << std::endl;
			std::cout << "Preint:\n" << frame_cur->getPreintergration() << std::endl;
		}

		all_frames_.push_back(frame_cur);
	}

	bool tryInit(VectorXd &result, bool report = false, bool verbose = false)
	{
		bool intialed = Optimizer::initIMU(all_frames_, result, report, verbose);
		return intialed;
	}

    const std::vector<Frame::Ptr> &frames() const { return all_frames_; }

    bool checkFrameAdd(double timestamp, double duration)
    {
        return (all_frames_.empty() || (timestamp - all_frames_.back()->timestamp_ > duration));
    }

private:
	const EuRocDataReader * dataset_;
	AbstractCamera::Ptr camera_;
	size_t imu_idx_;
	std::vector<Frame::Ptr> all_frames_;
};

void loadTrajectory(std::string trajectory_file, std::vector<Sophus::SE3d> &poses, std::vector<double> &timestamps)
{
    std::ifstream file_stream;
    file_stream.open(trajectory_file.c_str());
    LOG_ASSERT(file_stream.is_open()) << "Can not open trajectory file: " << trajectory_file;
    while (!file_stream.eof()) {
        std::string s;
        getline(file_stream, s);
        if (!s.empty()) {
            std::stringstream ss;
            ss << s;
            double time;
            ss >> time;
            timestamps.push_back(time);

            std::vector<double> ground_truth(7);
            for (int i = 0; i < 7; ++i)
                ss >> ground_truth[i];

            Quaterniond quat_wc(ground_truth[6], ground_truth[3], ground_truth[4], ground_truth[5]);
            Vector3d tran_wc(ground_truth[0], ground_truth[1], ground_truth[2]);
            poses.emplace_back(Sophus::SE3d(quat_wc, tran_wc));
        }
    }
    file_stream.close();
}

int main(int argc, char *argv[])
{
	//FLAGS_alsologtostderr = true;
	//FLAGS_colorlogtostderr = true;
	//FLAGS_log_prefix = false;
	google::InitGoogleLogging(argv[0]);
	LOG_ASSERT(argc == 4 || argc == 5) << "\n Usage : ./test_perintegration_init calib_file config_file dataset_path trajectory";

    bool pose_from_groundtruth = true;

    std::vector<Sophus::SE3d> poses;
    std::vector<double> timestamps;
    if (argc == 5)
    {
        loadTrajectory(argv[4], poses, timestamps);

        LOG_ASSERT(poses.size() > 0) << "Empty pose from trajectory file: " << argv[4];
        pose_from_groundtruth = false;
    }

	ssvo::Config::file_name_ = argv[2];

	IMUPara::setMeasCov(2.0000e-3, 1.6968e-04);
	IMUPara::setBiasCov(3.0000e-3, 1.9393e-05);
	IMUPara::setGravity(47.376, 0.0);
	Vector3d gyro_bias(0.0, 0.0, 0.0);// (-0.002571, 0.021269, 0.076861);
	Vector3d acc_bias(0.0, 0.0, 0.0);// -0.022996, 0.125896, 0.057076);

	EuRocDataReader dataset(argv[3]);
	AbstractCamera::Ptr camera = PinholeCamera::create(argv[1]);
	IMUInitializer initialier(&dataset, camera);

	std::ofstream out_file("imu_bias.txt");
	out_file << std::fixed << std::setprecision(7);
	LOG_ASSERT(out_file.is_open()) << "Error in open out file!";

	ssvo::Timer<std::milli> timer;

    const Sophus::SE3d Tbc(camera->T_BC());
	size_t imu_idx = 0;
	const size_t N = dataset.leftImageSize();
    double keyframe_duration = 1.0;
    const size_t max_frame = 50;

    cv::Mat image;
	for (size_t i = 0; i < N; i++)
	{
        double timestamp;
        const EuRocDataReader::Image image_data = dataset.leftImage(i);
		std::cout << std::fixed << std::setprecision(7);
		image = cv::imread(image_data.path, CV_LOAD_IMAGE_UNCHANGED);
		if (image.empty()) continue;
        timestamp = image_data.timestamp;

        //! use the next for
        if (!pose_from_groundtruth) break;

        Sophus::SE3d Twc, Twb;
        double groundtruth_timestamp;
        bool ok = initialier.getTruthPose(timestamp, groundtruth_timestamp, Twb);
        Twc = Twb * Tbc;
        if (!ok) continue;

        if (!initialier.checkFrameAdd(timestamp, keyframe_duration))
            continue;

        std::cout << "=== Add Frame at " << i << ", time: " << timestamp << " gt: " << groundtruth_timestamp  << std::endl;
        initialier.addNewFrame(image, Twc, timestamp, 1e-2, true);

		timer.start();
        VectorXd result;
        bool succeed = initialier.tryInit(result, true, false);
		double dt = timer.stop();

		std::cout << "result: " << result.transpose() << std::endl;

        //! log to file
		size_t value = result.size();
		result.conservativeResize(10);
		result.tail(10 - value).setZero();

		out_file << timestamp << " " << result .transpose() << " " << dt << " " << (int)succeed << " " << initialier.frames().size() << std::endl;
	}

    keyframe_duration = 1.0;
    bool align_to_gt = true;
    bool use_gt = false;
    Sophus::SE3d T_wofgt_from_wofc;
    Sophus::SE3d Twb_gt_last;
    const size_t M = poses.size();
    for (size_t i = 0; i < M; i++)
    {
        double timestamp = timestamps[i];
        Sophus::SE3d Twc = poses[i];

        if (!initialier.checkFrameAdd(timestamp, keyframe_duration))
            continue;

        Sophus::SE3d Twb_gt;
        double groundtruth_timestamp = -1;
        bool ok = initialier.getTruthPose(timestamp, groundtruth_timestamp, Twb_gt);
        if (!ok) continue;

        if (align_to_gt && initialier.frames().empty())
        {
            align_to_gt = false;
            Sophus::SO3d R_wofgt_cam = Twb_gt.so3() * Tbc.so3();
            Sophus::SO3d R_wofgt_from_wofc = R_wofgt_cam * Twc.so3().inverse();
            T_wofgt_from_wofc.so3() = R_wofgt_from_wofc;
            T_wofgt_from_wofc.translation().setZero();
        }

        std::cout << "=== Add Frame at " << i << ", time: " << timestamp << " gt: " << groundtruth_timestamp << std::endl;
        
        if(!use_gt)
            initialier.addNewFrame(image, T_wofgt_from_wofc * Twc, timestamp, 0.0, true);
        else
            initialier.addNewFrame(image, Twb_gt * Tbc, timestamp, 0.0, true);

        Sophus::SE3d Tij = Twb_gt_last.inverse() * Twb_gt;
        Quaterniond qij = Tij.unit_quaternion();
        std::cout << "gt Tij[t, q]: " << Tij.translation().transpose() << ", (" << qij.x() << ", " << qij.y() << ", " << qij.z() << ", " << qij.w() << ")" << std::endl;
        Twb_gt_last = Twb_gt;

        timer.start();
        VectorXd result;
        bool succeed = initialier.tryInit(result, true, false);
        double dt = timer.stop();

        std::cout << "result: " << result.transpose() << std::endl;

        //! log to file
        size_t value = result.size();
        result.conservativeResize(10);
        result.tail(10 - value).setZero();

        out_file << timestamp << " " << result.transpose() << " " << dt << " " << (int)succeed << " " << initialier.frames().size() << std::endl;
    }

	out_file.close();

	getchar();
	getchar();

	return 0;
}