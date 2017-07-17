#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include <vector>
#include <memory>
#include <assert.h>

#include <opencv2/core.hpp>

#include "feature_detector.hpp"

namespace ssvo{

class Frame;

typedef std::shared_ptr<Frame> FramePtr;

class Frame
{
public:
    Frame(const cv::Mat& img, const double timestamp, cv::Mat& K, cv::Mat& dist_coef);

    static int createPyramid(const cv::Mat& img, std::vector<cv::Mat>& img_pyr, const uint16_t nlevels = 4, const cv::Size min_size = cv::Size(40, 40));

    int detectFeatures(FastDetector& detector);

    inline long int ID(){return id_;};

    cv::Mat K(){return K_;}

public:
    static long int frameId;

    ImgPyr img_pyr_;
    std::vector<cv::KeyPoint> kps_;
    std::vector<cv::Point2f> pts_;    //! tracked

private:
    const long int id_;
    const double timestamp_;

    int pyr_levels_;
    cv::Mat K_;
    float fx, fy, cx, cy;
    float k1, k2, p1, p2;
    cv::Mat dist_coef_;

};

}

#endif