#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <include/config.hpp>

#include "frame.hpp"
#include "keyframe.hpp"
#include "utils.hpp"

namespace ssvo {

uint64_t Frame::next_id_ = 0;
const cv::Size Frame::optical_win_size_ = cv::Size(21,21);
float Frame::light_affine_a_ = 1.0f;
float Frame::light_affine_b_ = 0.0f;

bool Frame::isInit_ = false;
int Frame::nlevels_;
double Frame::scale_factor_;
double Frame::log_scale_factor_;
std::vector<double> Frame::scale_factors_;
std::vector<double> Frame::inv_scale_factors_;
std::vector<double> Frame::level_sigma2_;
std::vector<double> Frame::inv_level_sigma2_;

Frame::Frame(const cv::Mat &img, const double timestamp, const AbstractCamera::Ptr &cam) :
    id_(next_id_++), timestamp_(timestamp), cam_(cam)
{
    Tcw_ = SE3d(Matrix3d::Identity(), Vector3d::Zero());
    Twc_ = Tcw_.inverse();

    LOG_ASSERT(isInit_) << "Please init Frame's scale parameters first!";

    //! copy to image pyramid
    img_pyr_.resize(nlevels_);
    for(int i = 0; i < nlevels_; i++)
    {
        float scale = inv_scale_factors_[i];
        cv::Size sz(cvRound((float)img.cols*scale), cvRound((float)img.rows*scale));

        if(0 == i)
        {
            img_pyr_[i] = img.clone();
        }
        else
        {
            img_pyr_[i] = cv::Mat(sz, img.type());
            cv::resize(img_pyr_[i-1], img_pyr_[i], sz, 0, 0, cv::INTER_LINEAR);
        }
    }

    fts_.reserve(500);
    mpts_.reserve(500);
    seeds_.reserve(500);
}

Frame::Frame(const ImgPyr &img_pyr, const uint64_t id, const double timestamp, const AbstractCamera::Ptr &cam) :
    id_(id), timestamp_(timestamp), cam_(cam), img_pyr_(img_pyr),
    Tcw_(SE3d(Matrix3d::Identity(), Vector3d::Zero())), Twc_(Tcw_.inverse())
{}

void Frame::initScaleParameters(const FastDetector::Ptr &fast)
{
    nlevels_ = fast->getNLevels();
    scale_factor_ = fast->getScaleFactor();
    log_scale_factor_ = fast->getLogScaleFactor();

    scale_factors_ = fast->getScaleFactors();
    inv_scale_factors_ = fast->getInvScaleFactors();

    level_sigma2_ = fast->getLevelSigma2();
    inv_level_sigma2_ = fast->getInvLevelSigma2();

    isInit_ = true;
}

const ImgPyr Frame::images() const
{
    return img_pyr_;
}

const cv::Mat Frame::getImage(int level) const
{
    LOG_ASSERT(level < (int) img_pyr_.size()) << "Error level: " << level;
    return img_pyr_[level];
}

const ImgPyr Frame::opticalImages()
{
    //! create pyramid for optical flow
    if(optical_pyr_.empty())
        cv::buildOpticalFlowPyramid(images()[0], optical_pyr_, optical_win_size_, 4, false);
    return optical_pyr_;
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

bool Frame::isVisiable(const Vector3d &xyz_w, const int border)
{
    SE3d Tcw;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Tcw = Tcw_;
    }
    const Vector3d xyz_c = Tcw * xyz_w;
    if(xyz_c[2] < 0.0f)
        return false;

    Vector2d px = cam_->project(xyz_c);
    return cam_->isInFrame(px.cast<int>(), border);
}

bool Frame::addMapPointFeatureMatch(const MapPoint::Ptr &mpt, const Feature::Ptr &ft)
{
    if(!ft || !mpt) return false;

    const size_t N = fts_.size();
    fts_.push_back(ft);
    mpts_.push_back(mpt);
    seeds_.push_back(nullptr);

    return mpt_matches_.insert(N).second;
}

bool Frame::addSeedFeatureMatch(const Seed::Ptr &seed, const Feature::Ptr &ft)
{
    if(!ft || !seed) return false;

    const size_t N = fts_.size();
    fts_.push_back(ft);
    mpts_.push_back(nullptr);
    seeds_.push_back(seed);

    return seed_matches_.insert(N).second;
}

size_t Frame::getMapPointMatchSize()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return mpt_matches_.size();
}

size_t Frame::getSeedMatchSize()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return seed_matches_.size();
}

std::vector<size_t> Frame::getMapPointMatchIndices()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return std::vector<size_t>(mpt_matches_.begin(), mpt_matches_.end());
}

std::vector<size_t> Frame::getSeedMatchIndices()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return std::vector<size_t>(seed_matches_.begin(), seed_matches_.end());;
}

std::unordered_map<MapPoint::Ptr, Feature::Ptr> Frame::getMapPointFeatureMatches()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    std::unordered_map<MapPoint::Ptr, Feature::Ptr> map_fts;
    for(const size_t &idx : mpt_matches_)
        map_fts.emplace(mpts_[idx], fts_[idx]);

    return map_fts;
}

std::unordered_map<Seed::Ptr, Feature::Ptr> Frame::getSeedFeatureMatches()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    std::unordered_map<Seed::Ptr, Feature::Ptr> seed_fts;
    for(const size_t &idx : seed_matches_)
        seed_fts.emplace(seeds_[idx], fts_[idx]);

    return seed_fts;
}

bool Frame::removeMapPointMatchByIndex(const size_t &idx)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    const auto it = mpt_matches_.find(idx);
    if(it == mpt_matches_.end()) return false;
    mpt_matches_.erase(it);
    return true;
}

bool Frame::removeSeedMatchByIndex(const size_t &idx)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    const auto it = seed_matches_.find(idx);
    if(it == seed_matches_.end()) return false;
    seed_matches_.erase(it);
    return true;
}

std::vector<Feature::Ptr> Frame::getFeatures()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return fts_;
}

std::vector<MapPoint::Ptr> Frame::getMapPoints()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return mpts_;
}

std::vector<Seed::Ptr> Frame::getSeeds()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return seeds_;
}

bool Frame::getSceneDepth(double &depth_median, double &depth_min)
{
    SE3d Tcw = this->Tcw();

    std::vector<MapPoint::Ptr> mpts = getMapPoints();

    std::vector<double> depth_vec;
    depth_vec.reserve(mpts.size());

    depth_min = std::numeric_limits<double>::max();
    for(const MapPoint::Ptr &mpt : mpts)
    {
        if(mpt == nullptr)
            continue;
        const Vector3d p =  Tcw * mpt->pose();
        depth_vec.push_back(p[2]);
        depth_min = fmin(depth_min, p[2]);
    }

    if(depth_vec.empty())
        return false;

    depth_median = utils::getMedian(depth_vec);
    return true;
}

std::map<KeyFrame::Ptr, int> Frame::getOverLapKeyFrames()
{
    std::vector<MapPoint::Ptr> mpts = getMapPoints();

    std::map<KeyFrame::Ptr, int> overlap_kfs;

    for(const MapPoint::Ptr &mpt : mpts)
    {
        if(mpt == nullptr)
            continue;

        const auto &obs = mpt->getObservations();
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