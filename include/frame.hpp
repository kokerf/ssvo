#ifndef _FRAME_HPP_
#define _FRAME_HPP_

#include <vector>
#include <memory>
#include <assert.h>

#include <opencv2/core.hpp>

#include "camera.hpp"
#include "feature_detector.hpp"
#include "global.hpp"

namespace ssvo{

typedef std::vector<Feature*> Features;
typedef std::vector<MapPoint*> MapPoints;

class Frame
{
public:
    Frame(const cv::Mat& img, const double timestamp, CameraPtr cam);

    static int createPyramid(const cv::Mat& img, std::vector<cv::Mat>& img_pyr, const uint16_t nlevels = 4, const cv::Size min_size = cv::Size(40, 40));

    void addFeature(Feature* ft);

    inline long int ID(){return id_;};

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static long int frame_id_;
    const long int id_;
    const double timestamp_;

    int pyr_levels_;

    CameraPtr cam_;
    ImgPyr img_pyr_;
    Features fts_;
    MapPoints mps_;

};

}

#endif