#include "map_point.hpp"
#include "keyframe.hpp"

namespace ssvo
{

uint64_t MapPoint::next_id_ = 0;

MapPoint::MapPoint(const Vector3d &p) :
        id_(next_id_++), last_structure_optimal_(0), pose_(p), type_(SEED),
        min_distance_(0.0), max_distance_(0.0), refKF_(nullptr), found_cunter_(1), visiable_cunter_(1)
{
}

MapPoint::Type MapPoint::type()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return type_;
}

void MapPoint::resetType(MapPoint::Type type)
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    type_ = type;
}

void MapPoint::setBad()
{
    std::unordered_map<KeyFrame::Ptr, size_t > obs;
    {
        std::lock_guard<std::mutex> lock(mutex_obs_);
        type_ = BAD;
        obs = obs_;
    }

    for(const auto &it : obs)
        it.first->removeMapPointMatchByIndex(it.second);

    for(const auto &it : obs)
        it.first->updateConnections();

    {
        std::lock_guard<std::mutex> lock(mutex_obs_);
        obs_.clear();
    }
}

bool MapPoint::isBad()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return type_ == BAD;
}

KeyFrame::Ptr MapPoint::getReferenceKeyFrame()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return refKF_;
}

void MapPoint::addObservation(const KeyFrame::Ptr &kf, const size_t &idx)
{
    LOG_ASSERT(kf) << " Error input kf: " << kf;

    std::lock_guard<std::mutex> lock(mutex_obs_);
    LOG_ASSERT(type_ != BAD) << " Error to use a BAD MapPoint!";

    if(refKF_ == nullptr)
        refKF_ = kf;
    obs_.emplace(kf, idx);
}

//! it do not change the connections of keyframe
bool MapPoint::fusion(const MapPoint::Ptr &mpt)
{
    const auto obs = mpt->getObservations();
    bool update = false;
    {
        std::lock_guard<std::mutex> lock(mutex_obs_);
        found_cunter_ += mpt->getFound();
        visiable_cunter_ += mpt->getVisible();

        for(const auto &it : obs)
        {
            if(obs_.count(it.first) == 0)
            {
                obs_.insert(it);
                update = true;
            }
        }
    }

    mpt->setBad();

    if(update)
        updateViewAndDepth();

    return true;
}

int MapPoint::observations()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return (int)obs_.size();
}

//! should update connections for keyframe
bool MapPoint::removeObservation(const KeyFrame::Ptr &kf)
{
    {
        std::lock_guard<std::mutex> lock(mutex_obs_);
        const auto it = obs_.find(kf);
        if(it == obs_.end())
            return false;

//        LOG(INFO) << " Remove obs, mpt: " << id_ << " kf: " << kf->id_ << " size: " << obs_.size();

        const size_t &idx = it->second;
        kf->removeMapPointMatchByIndex(idx);
        obs_.erase(kf);
        if(obs_.empty())
        {
            type_ = BAD;
            return true;
        }

    }

    KeyFrame::Ptr ref_kf;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        ref_kf = refKF_;
    }

    if(kf == ref_kf)
        updateRefKF();

    updateViewAndDepth();

    return true;
}

void MapPoint::updateRefKF()
{
    uint64_t min_id = std::numeric_limits<uint64_t>::max();
    KeyFrame::Ptr ref_kf;
    {
        std::lock_guard<std::mutex> lock(mutex_obs_);
        for(const auto &item : obs_)
        {
            if(item.first->id_ < min_id)
            {
                min_id = item.first->id_;
                ref_kf = item.first;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        refKF_ = ref_kf;
    }
}

std::map<KeyFrame::Ptr, size_t> MapPoint::getObservations()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return std::map<KeyFrame::Ptr, size_t >(obs_.begin(), obs_.end());
}


size_t MapPoint::getFeatureIndex(const KeyFrame::Ptr &kf)
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    const auto it = obs_.find(kf);
    if(it != obs_.end())
        return it->second;
    else
        return -1;
}

void MapPoint::updateViewAndDepth()
{
    if(isBad())
        return;

    {
        std::lock_guard<std::mutex> lock(mutex_obs_);

        if(obs_.empty())
            return;

        Vector3d normal = Vector3d::Zero();
        int n = 0;
        for(const auto &obs : obs_)
        {
            Vector3d Ow = obs.first->pose().translation();
            Vector3d obs_dir((Ow - pose_).normalized());
            normal = normal + obs_dir;
            n++;
        }
        obs_dir_ = normal / n;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        Vector3d ref_obs_dir = refKF_->pose().translation() - pose_;

        const double dist = ref_obs_dir.norm();
        const size_t idx = getFeatureIndex(refKF_);
        const Feature::Ptr ft = refKF_->getFeatureByIndex(idx);
        const double level_scale = Frame::scale_factors_.at(ft->corner_.level);
        const double max_scale = Frame::scale_factors_.back();

        max_distance_ = dist * level_scale; //! regard it is top level, we may obsevere the point if we go closer
        min_distance_ = max_distance_ / max_scale;
    }
}

void MapPoint::computeDistinctiveDescriptor()
{
    if(isBad())
        return;

    std::unordered_map<KeyFrame::Ptr, size_t> obs_temp;
    {
        std::lock_guard<std::mutex> lock(mutex_obs_);
        obs_temp = obs_;
    }

    if(obs_temp.empty())
        return;

    std::vector<cv::Mat> descriptors;
    descriptors.reserve(obs_temp.size());

    for(const auto &obv : obs_temp)
    {
        const KeyFrame::Ptr kf = obv.first;
        if(kf->isBad()) continue;
        descriptors.push_back(kf->descriptors_.at(obv.second));
    }

    if(descriptors.empty())
        return;

    const size_t N = descriptors.size();

    std::vector<std::vector<int>> dist(N, std::vector<int>(N));
    for(size_t i=0;i<N;i++)
    {
        dist[i][i]=0;
        for(size_t j=i+1;j<N;j++)
        {
            int distij = DBoW3::DescManip::distance_8uc1(descriptors[i],descriptors[j]);
            dist[i][j]=distij;
            dist[j][i]=distij;
        }
    }

    int best_median = std::numeric_limits<int>::max();
    int best_idx = 0;

    for(size_t i=0;i<N;i++)
    {
        std::sort(dist[i].begin(), dist[i].end());
        int median = dist[i][0.5*(N-1)];

        if(median<best_median)
        {
            best_median = median;
            best_idx = i;
        }
    }

    {
        std::lock_guard<std::mutex> lock(mutex_obs_);
        descriptor_ = descriptors.at(best_idx).clone();
    }
}

cv::Mat MapPoint::descriptor()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return descriptor_.clone();
}

double MapPoint::getMinDistanceInvariance()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return 0.8f * min_distance_;
}

double MapPoint::getMaxDistanceInvariance()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    return 1.2f * max_distance_;
}

Vector3d MapPoint::getMeanViewDirection()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return obs_dir_;
}

int MapPoint::predictScale(const double dist, const int max_level)
{
    double ratio;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        ratio = max_distance_ / dist;
    }

    int scale = round(log(ratio) / Frame::log_scale_factor_);
    if(scale < 0)
        scale = 0;
    else if(scale > max_level)
        scale = max_level;

    return scale;
}

int MapPoint::predictScale(const double dist_ref, const double dist_cur, const int level_ref, const int max_level)
{
    double ratio = dist_ref * Frame::scale_factors_.at(level_ref) / dist_cur;

    int scale = round(log(ratio) / Frame::log_scale_factor_);
    if(scale < 0)
        scale = 0;
    else if(scale > max_level)
        scale = max_level;

    return scale;
}

void MapPoint::increaseFound(int n)
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    found_cunter_ += n;
}

void MapPoint::increaseVisible(int n)
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    visiable_cunter_ += n;
}

uint64_t MapPoint::getFound()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return found_cunter_;
}

uint64_t MapPoint::getVisible()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return visiable_cunter_;
}

double MapPoint::getFoundRatio()
{
    std::lock_guard<std::mutex> lock(mutex_obs_);
    return static_cast<double>(found_cunter_)/visiable_cunter_;
}

bool MapPoint::getCloseViewObs(const Frame::Ptr &frame, KeyFrame::Ptr &keyframe, int &level)
{
    std::unordered_map<KeyFrame::Ptr, size_t> obs;
    Vector3d obs_dir;
    {
        std::lock_guard<std::mutex> lock(mutex_obs_);
        if(type_ == BAD)
            return false;
        // TODO 这里可能还有问题，bad 的 mpt没有被删除？
        LOG_ASSERT(!obs_.empty()) << " Map point is invalid!";
        obs = obs_;
        obs_dir = obs_dir_;
    }

    //! 1. scale invariance region check
    Vector3d frame_obs_dir;
    {
        std::lock_guard<std::mutex> lock(mutex_pose_);
        frame_obs_dir = frame->pose().translation() - pose_;
    }
    const double dist = frame_obs_dir.norm();
    if(dist < getMinDistanceInvariance() || dist > getMaxDistanceInvariance())
        return false;

    //! 2. observation view check
    frame_obs_dir.normalize();
    if(frame_obs_dir.dot(obs_dir) < 0.5)
        return false;

    //! 3. find close view keyframe
    Vector3d frame_dir(frame->ray().normalized());

    double max_cos_angle = 0.0;
    for(const auto &item : obs)
    {
        Vector3d kf_dir(item.first->ray().normalized());
        double view_cos_angle = kf_dir.dot(frame_dir);

        //! find min angle
        if(view_cos_angle < max_cos_angle)
            continue;

        max_cos_angle = view_cos_angle;
        keyframe = item.first;
    }

    if(max_cos_angle < 0.5f)
        return false;

    level = predictScale(dist, frame->nlevels_-1);

    return true;
}

}
