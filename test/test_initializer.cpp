#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature_detector.hpp"
#include "initializer.hpp"
#include "config.hpp"

std::string ssvo::Config::FileName;

int main(int argc, char const *argv[])
{
    if (argc != 4) {
        std::cout << "Usge: ./test_initializer image0 image1 configflie" << std::endl;
    }

    cv::Mat ref_img = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat cur_img = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    ssvo::Config::FileName = std::string(argv[3]);

    std::cout << "Config: " << ssvo::Config::initMinCorners() << std::endl;

    if(ref_img.empty())
    {
        std::cout << "Can not open:" << argv[1] << std::endl;
        return -1;
    }

    if(cur_img.empty())
    {
        std::cout << "Can not open:" << argv[2] << std::endl;
        return -1;
    }

    cv::Mat K = ssvo::Config::cameraIntrinsic();
    cv::Mat DistCoef = ssvo::Config::cameraDistCoef();
    ssvo::Frame frame1(ref_img, 0, K, DistCoef);
    ssvo::Frame frame2(cur_img, 0, K, DistCoef);

    ssvo::FastDetector fast_detector(1000, 3);
    frame1.detectFeatures(fast_detector);

    ssvo::Initializer initializer(std::make_shared<ssvo::Frame>(frame1));

    bool succeed = initializer.initialize(std::make_shared<ssvo::Frame>(frame2));


    return 0;
}