#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <include/config.hpp>

#include "frame.hpp"
#include "keyframe.hpp"
#include "utils.hpp"

namespace ssvo {

uint64_t Frame::next_id_ = 0;
const cv::Size Frame::optical_win_size_ = cv::Size(21,21);

Frame::Frame(const cv::Mat &img, const double timestamp, const Camera::Ptr &cam) :
    id_(next_id_++), timestamp_(timestamp), cam_(cam), max_level_(Config::imageTopLevel())
{
    Tcw_ = SE3d(Matrix3d::Identity(), Vector3d::Zero());
    Twc_ = Tcw_.inverse();

//    utils::createPyramid(img, img_pyr_, nlevels_);
    //! create pyramid for optical flow
    cv::buildOpticalFlowPyramid(img, optical_pyr_, optical_win_size_, max_level_, false);
    LOG_ASSERT(max_level_ == optical_pyr_.size()-1) << "The pyramid level is unsuitable! maxlevel should be " << optical_pyr_.size()-1;

    //! copy to image pyramid
    img_pyr_.resize(optical_pyr_.size());
    for(size_t i = 0; i < optical_pyr_.size(); i++)
        optical_pyr_[i].copyTo(img_pyr_[i]);

}

Frame::Frame(const ImgPyr &img_pyr, const uint64_t id, const double timestamp, const Camera::Ptr &cam) :
    id_(id), timestamp_(timestamp), cam_(cam), max_level_(img_pyr.size()), img_pyr_(img_pyr),
    Tcw_(SE3d(Matrix3d::Identity(), Vector3d::Zero())), Twc_(Tcw_.inverse())
{}

const ImgPyr Frame::images() const
{
    return img_pyr_;
}

const ImgPyr Frame::opticalImages() const
{
    return optical_pyr_;
}

const cv::Mat Frame::getImage(int level) const
{
    LOG_ASSERT(level < img_pyr_.size()) << "Error level: " << level;
    return img_pyr_[level];
}

SE3d Frame::Tcw()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Tcw_;
}

SE3d Frame::Twc()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Twc_;
}

SE3d Frame::pose()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Twc_;
}

Vector3d Frame::ray()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return Dw_;
}

void Frame::setPose(const SE3d& pose)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Twc_ = pose;
    Tcw_ = Twc_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);
}

void Frame::setPose(const Matrix3d& R, const Vector3d& t)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Twc_ = SE3d(R, t);
    Tcw_ = Twc_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);
}

void Frame::setTcw(const SE3d &Tcw)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    Tcw_ = Tcw;
    Twc_ = Tcw_.inverse();
    Dw_ = Tcw_.rotationMatrix().determinant() * Tcw_.rotationMatrix().col(2);

}

bool Frame::isVisiable(const Vector3d &xyz_w)
{
    SE3d Tcw;
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

bool Frame::getSceneDepth(double &depth_mean, double &depth_min)
{
    SE3d Tcw;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Tcw = Tcw_;
    }
    Features fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        fts = fts_;
    }

    std::vector<double> depth_vec;
    depth_vec.reserve(fts.size());

    depth_min = std::numeric_limits<double>::max();
    for(const Feature::Ptr &ft : fts)
    {
        if(ft->mpt == nullptr)
            continue;

        const Vector3d p =  Tcw * ft->mpt->pose();
        depth_vec.push_back(p[2]);
        depth_min = fmin(depth_min, p[2]);
    }

    if(depth_vec.empty())
        return false;

    depth_mean = utils::getMedian(depth_vec);
    return true;
}

std::map<KeyFrame::Ptr, int> Frame::getOverLapKeyFrames()
{
    Features fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        fts = fts_;
    }

    std::map<KeyFrame::Ptr, int> overlap_kfs;

    for(const Feature::Ptr &ft : fts)
    {
        if(ft->mpt == nullptr)
            continue;

        std::map<KeyFrame::Ptr, Feature::Ptr> obs = ft->mpt->getObservations();
        for(const auto &item : obs)
        {
            auto it = overlap_kfs.find(item.first);
            if(it != overlap_kfs.end())
                it->second++;
            else
                overlap_kfs.insert(std::make_pair(item.first, 1));
        }
    }

    return overlap_kfs;
}

}