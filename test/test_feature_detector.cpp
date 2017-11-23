#include <iostream>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "feature_detector.hpp"

using namespace cv;
using namespace ssvo;

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size);

int main(int argc, char const *argv[])
{
    if(argc != 3)
    {
        std::cout << "Usge: ./test_feature_detector image configfile" << std::endl;
    }

    google::InitGoogleLogging(argv[0]);

    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    LOG_IF(FATAL, image.empty()) << "Can not open:" << argv[1];

    Config::FileName = std::string(argv[2]);
    int width = Config::imageWidth();
    int height = Config::imageHeight();
    int level = Config::imageTopLevel();
    int image_border = Config::imageBorder();
    int grid_size = Config::gridSize();
    int grid_min_size = Config::gridMinSize();
    int fast_max_threshold = Config::fastMaxThreshold();
    int fast_min_threshold = Config::fastMinThreshold();
    double fast_min_eigen = Config::fastMinEigen();

    std::vector<cv::Mat> image_pyramid;
    int top_level = computePyramid(image, image_pyramid, 2, 4, cv::Size(40, 40));

    std::vector<Corner> corners, old_corners;
    FastDetector::Ptr fast_detector = FastDetector::create(width, height, image_border, level+1, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);

    LOG(WARNING) << "=== This is a FAST corner detector demo ===";
    const int n_trials = 1000;
    double time_accumulator = 0;
    for(int i = 0; i < n_trials; ++i)
    {
        double t = (double)cv::getTickCount();
        fast_detector->detect(image_pyramid, corners, old_corners, 100, fast_min_eigen);
        time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
        LOG_EVERY_N(WARNING, n_trials/20) << " i: " << i << ", corners: " << corners.size();
    }
    LOG(WARNING) << " took " <<  time_accumulator/((double)n_trials)*1000.0
              << " ms (average over " << n_trials << " trials)." << std::endl;

    cv::Mat kps_img;
    std::vector<cv::KeyPoint> keypoints;
    std::for_each(corners.begin(), corners.end(), [&](Corner corner){
        cv::KeyPoint kp(corner.x, corner.y, 0);
        keypoints.push_back(kp);
    });
    cv::drawKeypoints(image, keypoints, kps_img);

    fast_detector->drawGrid(kps_img, kps_img);
    cv::imshow("KeyPoints detectByImage", kps_img);
    cv::waitKey(0);

    return 0;
}

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size)
{
    LOG_ASSERT(scale_factor > 1.0);
    LOG_ASSERT(!image.empty());

    image_pyramid.resize(level + 1);

    image_pyramid[0] = image.clone();
    for(int i = 1; i <= level; ++i)
    {
        cv::Size size(round(image_pyramid[i - 1].cols / scale_factor), round(image_pyramid[i - 1].rows / scale_factor));

        if(size.height < min_size.height || size.width < min_size.width)
        {
            image_pyramid.resize(level);
            return level-1;
        }

        cv::resize(image_pyramid[i - 1], image_pyramid[i], size, 0, 0, cv::INTER_LINEAR);
    }
    return level;
}