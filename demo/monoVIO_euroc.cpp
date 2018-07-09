#include "system.hpp"
#include "dataset.hpp"
#include "time_tracing.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    LOG_ASSERT(argc == 4) << "\n Usage : ./monoVO_dataset config_file calib_file dataset_path";

    System vo(argv[1], argv[2]);

    EuRocDataReader dataset(argv[3]);

    IMUPara::setMeasCov(2.0000e-3, 1.6968e-04);
    IMUPara::setBiasCov(3.0000e-3, 1.9393e-05);
    IMUPara::setGravity(47.376, 0.0);

    ssvo::Timer<std::micro> timer;
    size_t imu_idx = 0;
    bool first_frame = true;
    const size_t N = dataset.leftImageSize();
    for(size_t i = 100; i < N; i++)
    {
        const EuRocDataReader::Image image_data = dataset.leftImage(i);
        LOG(INFO) << "=== Load Image " << i << ": " << image_data.path << ", time: " << std::fixed <<std::setprecision(7)<< image_data.timestamp << std::endl;
        cv::Mat image = cv::imread(image_data.path, CV_LOAD_IMAGE_UNCHANGED);
        if(image.empty())
            continue;

        if (first_frame)
        {
            while (1)
            {
                EuRocDataReader::IMUData data = dataset.imu(imu_idx);
                if (data.timestamp >= image_data.timestamp)
                    break;

                imu_idx++;
            }
            first_frame = false;
        }

        std::vector<IMUData> imu_datas;
        while (1)
        {
            EuRocDataReader::IMUData data = dataset.imu(imu_idx);

            if (data.timestamp >= image_data.timestamp)
                break;

            IMUData imu_data(data.timestamp, Vector3d(data.gyro[0], data.gyro[1], data.gyro[2]), Vector3d(data.acc[0], data.acc[1], data.acc[2]));
            imu_datas.push_back(imu_data);

            imu_idx++;
        }

        timer.start();
        vo.process(image, image_data.timestamp, imu_datas);
        timer.stop();

        double time_process = timer.duration();

        double time_wait = 0;
        if(i < N -1)
            time_wait = (dataset.leftImage(i+1).timestamp - image_data.timestamp)*1e6;
        else
            time_wait = (image_data.timestamp - dataset.leftImage(i-1).timestamp)*1e6;

        if(time_process < time_wait)
            std::this_thread::sleep_for(std::chrono::microseconds((int)(time_wait - time_process)));
    }

    vo.saveTrajectoryTUM("trajectory.txt");
    vo.saveTrajectoryTUM("keyframe_trajectory.txt", 1);
    getchar();

    return 0;
}