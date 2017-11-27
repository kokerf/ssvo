#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <include/config.hpp>

#include "frame.hpp"
#include "utils.hpp"

namespace ssvo {

uint64_t Frame::next_id_ = 0;

Frame::Frame(const cv::Mat &img, const double timestamp, Camera::Ptr cam) :
    id_(next_id_++), timestamp_(timestamp), cam_(cam) {
    Tcw_ = Sophus::SE3d(Matrix3d::Identity(), Vector3d::Zero());
    Twc_ = Tcw_.inverse();

    utils::createPyramid(img, img_pyr_, Config::imageTopLevel() + 1);
}

Frame::Frame(const ImgPyr &img_pyr, const double timestamp, Camera::Ptr cam) :
    id_(next_id_++), timestamp_(timestamp), cam_(cam), img_pyr_(img_pyr),
    Tcw_(Sophus::SE3d(Matrix3d::Identity(), Vector3d::Zero())), Twc_(Tcw_.inverse()) {
}

Frame::Frame(const ImgPyr &img_pyr, const uint64_t id, const double timestamp, Camera::Ptr cam) :
    id_(id), timestamp_(timestamp), cam_(cam), img_pyr_(img_pyr),
    Tcw_(Sophus::SE3d(Matrix3d::Identity(), Vector3d::Zero())), Twc_(Tcw_.inverse()) {
}

const cv::Mat Frame::getImage(int level) const {
    LOG_ASSERT(level < img_pyr_.size()) << "Error level: " << level;
    return img_pyr_[level];
}

}