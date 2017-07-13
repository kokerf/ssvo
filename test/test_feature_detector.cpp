#include <iostream>
#include <opencv2/opencv.hpp>

#include "feature_detector.hpp"

using namespace cv;

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size);

int main(int argc, char const *argv[])
{
    if(argc != 2)
    {
        std::cout << "Usge: ./test_feature_detector image" << std::endl;
    }

    cv::Mat image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    if(image.empty())
    {
        std::cout << "Can not open:" << argv[1] << std::endl;
    }

    std::vector<cv::Mat> image_pyramid;
    int nlevels = 1 + computePyramid(image, image_pyramid, 2, 4, cv::Size(40, 40));

    std::vector<cv::KeyPoint> all_keypoints, old_keypoints;
    ssvo::FastDetector fast(1000, nlevels);

    fast(image_pyramid, old_keypoints, all_keypoints);

    const int n_trials = 1000;
    double time_accumulator = 0;
    for(int i = 0; i < n_trials; ++i)
    {
        all_keypoints.clear();
        double t = (double)cv::getTickCount();
        fast(image_pyramid, all_keypoints, old_keypoints);
        time_accumulator +=  ((cv::getTickCount() - t) / cv::getTickFrequency());
    }
    std::cout << "FAST Detector took " <<  time_accumulator/((double)n_trials)*1000.0
              << " ms (average over " << n_trials << " trials)." << std::endl;

    std::cout << "All : " << all_keypoints.size()
              << " old: " << old_keypoints.size()
              << " new: " << fast.new_coners_ << std::endl;

    cv::Mat kps_img;
    cv::drawKeypoints(image, all_keypoints, kps_img);
    cv::imshow("KeyPoints", kps_img);
    cv::waitKey(0);

    return 0;
}

int computePyramid(const cv::Mat& image, std::vector<cv::Mat>& image_pyramid, const float scale_factor, const uint16_t level, const cv::Size min_size)
{
    assert(scale_factor > 1.0);
    assert(!image.empty());

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