#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include <vector>
#include <memory>
#include <assert.h>

#include <opencv2/core.hpp>

#include "camera.hpp"
#include "feature_detector.hpp"

namespace ssvo{

//class Feature;

class Frame
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    Frame(const cv::Mat& img, const double timestamp, CameraPtr cam);

    static int createPyramid(const cv::Mat& img, std::vector<cv::Mat>& img_pyr, const uint16_t nlevels = 4, const cv::Size min_size = cv::Size(40, 40));

    int detectFeatures(FastDetector& detector);

    void addFeature();

    inline long int ID(){return id_;};

public:
    static long int frameId;

    CameraPtr cam_;
    ImgPyr img_pyr_;
    std::vector<cv::KeyPoint> kps_;
    std::vector<cv::Point2f> pts_;
    //std::vector<Feature> fts_;

private:
    const long int id_;
    const double timestamp_;

    int pyr_levels_;

};

typedef std::shared_ptr<Frame> FramePtr;

}

#endif