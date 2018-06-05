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

    ssvo::Timer<std::micro> timer;
    const size_t N = dataset.leftImageSize();
    size_t imu_idx = 0;
    for(size_t i = 0; i < N; i++)
    {
        const EuRocDataReader::Image image_data = dataset.leftImage(i);
        LOG(INFO) << "=== Load Image " << i << ": " << image_data.path << ", time: " << std::fixed <<std::setprecision(7)<< image_data.timestamp << std::endl;
        cv::Mat image = cv::imread(image_data.path, CV_LOAD_IMAGE_UNCHANGED);
        if(image.empty())
            continue;

        while (1)
        {
            EuRocDataReader::IMUData imu_data = dataset.imu(imu_idx);

            if (imu_data.timestamp >= image_data.timestamp)
                break;

            vo.addIMUData(ssvo::IMUData(imu_data.timestamp,
                                        Vector3d(imu_data.gyro[0], imu_data.gyro[1], imu_data.gyro[2]),
                                        Vector3d(imu_data.acc[0], imu_data.acc[1], imu_data.acc[2])));
            imu_idx++;
        }

        timer.start();
        vo.process(image, image_data.timestamp);
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
    getchar();

    return 0;
}