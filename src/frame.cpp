#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <include/config.hpp>

#include "frame.hpp"
#include "utils.hpp"

namespace ssvo {

uint64_t Frame::next_id_ = 0;

Frame::Frame(const cv::Mat &img, const double timestamp, Camera::Ptr cam) :
    id_(next_id_++), timestamp_(timestamp), cam_(cam), nlevels_(Config::imageTopLevel() + 1)
{
    Tcw_ = Sophus::SE3d(Matrix3d::Identity(), Vector3d::Zero());
    Twc_ = Tcw_.inverse();

    utils::createPyramid(img, img_pyr_, nlevels_);
}

Frame::Frame(const ImgPyr &img_pyr, const double timestamp, Camera::Ptr cam) :
    id_(next_id_++), timestamp_(timestamp), cam_(cam), nlevels_(img_pyr.size()), img_pyr_(img_pyr),
    Tcw_(Sophus::SE3d(Matrix3d::Identity(), Vector3d::Zero())), Twc_(Tcw_.inverse())
{}

Frame::Frame(const ImgPyr &img_pyr, const uint64_t id, const double timestamp, Camera::Ptr cam) :
    id_(id), timestamp_(timestamp), cam_(cam), nlevels_(img_pyr.size()), img_pyr_(img_pyr),
    Tcw_(Sophus::SE3d(Matrix3d::Identity(), Vector3d::Zero())), Twc_(Tcw_.inverse())
{}

const ImgPyr Frame::image() const
{
    return img_pyr_;
}

const cv::Mat Frame::getImage(int level) const
{
    LOG_ASSERT(level < img_pyr_.size()) << "Error level: " << level;
    return img_pyr_[level];
}

Sophus::SE3d Frame::Tcw()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Tcw_;
}

Sophus::SE3d Frame::pose()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Twc_;
}

Vector3d Frame::ray()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Dw_;
}

void Frame::setPose(const Sophus::SE3d& pose)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Twc_ = pose;
    Tcw_ = Twc_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);
}

void Frame::setPose(const Matrix3d& R, const Vector3d& t)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Twc_ = Sophus::SE3d(R, t);
    Tcw_ = Twc_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);
}

void Frame::setTcw(const Sophus::SE3d &Tcw)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Tcw_ = Tcw;
    Twc_ = Tcw_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

}

bool Frame::isVisiable(const Vector3d &xyz_w)
{
    Sophus::SE3d Tcw;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Tcw = Tcw_;
    }
    const Vector3d xyz_f = Tcw * xyz_w;
    if(xyz_f[2] < 0.0f)
        return false;

    Vector2d ft = cam_->project(xyz_f);
    return cam_->isInFrame(ft.cast<int>());
}

Features Frame::features()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return fts_;
}

std::vector<Feature::Ptr> Frame::getFeatures()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return std::vector<Feature::Ptr>(fts_.begin(), fts_.end());
}

void Frame::addFeature(const Feature::Ptr ft)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    fts_.push_back(ft);
}

}