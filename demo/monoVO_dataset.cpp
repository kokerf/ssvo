#include "system.hpp"
#include "dataset.hpp"
#include "time_tracing.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    LOG_ASSERT(argc == 4) << "\n Usage : ./monoVO_dataset config_file dataset_image_path  dataset_csv_file";

    System vo(argv[1]);
    EuRocDataReader dataset(argv[2], argv[3]);

    std::string image_name;
    double timestamp;
    ssvo::Timer<std::micro> timer;
    for(size_t i = 0; i < dataset.N; i++)
    {
        dataset.readItemByIndex(i, image_name, timestamp);
        LOG(INFO) << "=== Load Image " << i << ": " << image_name << ", time: " << std::fixed <<std::setprecision(7)<< timestamp << std::endl;
        cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_UNCHANGED);
        if(image.empty())
            continue;

        timer.start();
        vo.process(image, timestamp);
        timer.stop();

        double time_process = timer.duration();

        double time_wait;
        if(i < dataset.N -1)
            time_wait = (dataset.timestamps_[i+1] - timestamp)*1e6;
        else
            time_wait = (timestamp - dataset.timestamps_[i-1])*1e6;

        if(time_process < time_wait)
            usleep(time_wait-time_process);
    }

    vo.saveTrajectoryTUM("trajectory.txt");
    cv::waitKey(0);

    return 0;
}