#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <iomanip>
#include <memory>
#include "system.hpp"
#include "dataset.hpp"

using namespace ssvo;

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    LOG_ASSERT(argc == 4) << "\n Usage : ./monoVO config_file dataset_image_path  dataset_csv_file";

    System vo(argv[1]);
    EuRocDataReader dataset(argv[2], argv[3]);

    std::string image_name;
    double timestamp;
    for(size_t i = 0; i < dataset.N; i++)
    {
        dataset.readItemByIndex(i, image_name, timestamp);
        LOG(INFO) << "=== Load Image: " << image_name << ", time: " << std::fixed <<std::setprecision(7)<< timestamp << std::endl;
        cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_UNCHANGED);
        vo.process(image, timestamp);

        //cv::imshow("Image Show", image);
//        cv::waitKey(40);
    }

    return 0;
}