#ifndef _SSVO_BRIEF_HPP_
#define _SSVO_BRIEF_HPP_

#include <opencv2/core.hpp>

namespace ssvo
{

class BRIEF
{
public:
    enum
    {
        PATCH_SIZE = 31,
        HALF_PATCH_SIZE = 15,
        EDGE_THRESHOLD = 19,
    };

    BRIEF();

    void compute(const std::vector<cv::Mat> &images, const std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

    float IC_Angle(const cv::Mat &image, cv::Point2f pt, const std::vector<int> &u_max);

    void compute(const cv::KeyPoint &kpt, const cv::Mat &img, const cv::Point *pattern, uchar *desc);

private:

    std::vector<cv::Point> pattern_;

    std::vector<int> umax_;
};

}

#endif //_SSVO_BRIEF_HPP_
