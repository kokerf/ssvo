#include "config.hpp"
#include "brief.hpp"
#include "keyframe.hpp"

namespace ssvo{

uint64_t KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(const Frame::Ptr frame):
    Frame(frame->images(), next_id_++, frame->timestamp_, frame->cam_), frame_id_(frame->id_), isBad_(false)
{
    std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts = frame->getMapPointFeaturesMatched();
    std::unordered_map<Seed::Ptr, Feature::Ptr> seed_fts = frame->getSeedFeaturesMatched();
    N_ = mpt_fts.size() + seed_fts.size();
    mpts_.reserve(N_);
    seeds_.reserve(N_);
    fts_.reserve(N_);
    for(const auto &mpt_ft : mpt_fts)
    {
        mpts_matched_.insert(fts_.size());
        mpts_.push_back(mpt_ft.first);
        seeds_.push_back(nullptr);
        fts_.push_back(mpt_ft.second);
    }

    for(const auto &seed_ft : seed_fts)
    {
        seeds_matched_.insert(fts_.size());
        mpts_.push_back(nullptr);
        seeds_.push_back(seed_ft.first);
        fts_.push_back(seed_ft.second);
    }

    setRefKeyFrame(frame->getRefKeyFrame());
    setPose(frame->pose());

    grid_col_inv_ = static_cast<float>(GRID_COLS) / getImage(0).cols;
    grid_row_inv_ = static_cast<float>(GRID_ROWS) / getImage(0).rows;
}


size_t KeyFrame::N()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);

    return N_;
}

const Feature::Ptr& KeyFrame::getFeatureByIndex(const size_t &idx)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return fts_.at(idx);
}

const MapPoint::Ptr& KeyFrame::getMapPointByIndex(const size_t &idx)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return mpts_.at(idx);
}

bool KeyFrame::addMapPoint(const MapPoint::Ptr &mpt, const size_t &idx)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    if(idx < 0 || idx >= N_) return false;
    mpts_matched_.insert(idx);
    mpts_[idx] = mpt;
    return true;
}

bool KeyFrame::addSeedFeatureCreated(const Seed::Ptr &seed, const size_t &idx)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    if(idx < 0 || idx >= N_) return false;
    seeds_created_.insert(idx);
    seeds_[idx] = seed;
    return true;
}

bool KeyFrame::removeSeedCreateByIndex(const size_t &idx)
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    const auto it = seeds_created_.find(idx);
    if(it == seeds_created_.end()) return false;
    seeds_created_.erase(it);
    seeds_[idx] = nullptr;
    return true;
}

std::vector<size_t> KeyFrame::getSeedCreateIndices()
{
    std::lock_guard<std::mutex> lock(mutex_feature_);
    return std::vector<size_t>(seeds_created_.begin(), seeds_created_.end());
}

//bool KeyFrame::removeMapPointMatchByMapPoint(const MapPoint::Ptr &mpt)
//{
//    size_t idx = mpt->getFeatureIndex(shared_from_this());
//    if(idx < 0) return false;
//
//    return removeMapPointMatchByIndex(idx);
//}

void KeyFrame::detectFast(const FastDetector::Ptr &fast)
{
    const std::vector<Feature::Ptr> fts = getFeatures();

    Corners old_corners;
    old_corners.reserve(fts.size());
    for(const auto &ft : fts)
    {
        Corner &corner = ft->corner_;
//        if(corner.score < 0)
//            corner.score = FastDetector::shiTomasiScore(images()[corner.level], (int)std::round(corner.x), (int)std::round(corner.y));

        old_corners.emplace_back(corner);
    }

    Corners new_corners;
    fast->detect(images(), new_corners, old_corners, 1000);

    std::lock_guard<std::mutex> lock(mutex_feature_);
    {
        N_ = new_corners.size() + old_corners.size();

        fts_.reserve(N_);
        mpts_.resize(N_, nullptr);
        seeds_.resize(N_, nullptr);

        //! Add new features
        for(const Corner &corner : new_corners)
        {
            const Vector3d fn(cam_->lift(Vector2d(corner.x, corner.y)));
            fts_.emplace_back(Feature::create(corner, fn));
        }
    }

    assignFeaturesToGrid();
}

void KeyFrame::conputeDescriptor(const BRIEF::Ptr &brief)
{
    std::vector<cv::KeyPoint> kps; kps.reserve(N_);
    for(const Feature::Ptr & ft : fts_)
    {
        kps.emplace_back(cv::KeyPoint(ft->corner_.x, ft->corner_.y, 31, -1, 0, ft->corner_.level));
    }

    cv::Mat _descriptors;
    brief->compute(images(), kps, _descriptors);

    descriptors_.reserve(_descriptors.rows);
    for(int i = 0; i < _descriptors.rows; i++)
        descriptors_.push_back(_descriptors.row(i));
}

void KeyFrame::computeBoW(const DBoW3::Vocabulary& vocabulary)
{
    LOG_ASSERT(!descriptors_.empty()) << "Please use conputeDescriptor first!";
    if(bow_vec_.empty())
        vocabulary.transform(descriptors_, bow_vec_, feat_vec_, 4);
}

std::vector<size_t> KeyFrame::getFeaturesInArea(const float x,
                                                const float y,
                                                const float r,
                                                const int min_level,
                                                const int max_level)
{
    std::vector<size_t> indices;

    const int min_cell_x = std::max((int)std::floor((x-r)*grid_col_inv_), 0);
    if(min_cell_x >= GRID_COLS)
        return indices;

    const int max_cell_x = std::min((int)std::ceil(((x+r)*grid_col_inv_)), (int)GRID_COLS-1);
    if(max_cell_x < 0)
        return indices;

    const int min_cell_y = std::max((int)std::floor((y-r)*grid_row_inv_), 0);
    if(min_cell_y >= GRID_ROWS)
        return indices;

    const int max_cell_y = std::min((int)std::ceil(((x+r)*grid_row_inv_)), (int)GRID_ROWS-1);
    if(max_cell_y < 0)
        return indices;

    indices.reserve(N_);

    const float r2 = r*r;
    const bool check_level = (min_level >= 0) && (max_level >= 0);
    for(int iy = min_cell_y; iy <= max_cell_y; iy++)
        for(int ix = min_cell_x; ix <= max_cell_x; ix++)
        {
            const std::vector<size_t> cell_indices = grid_[iy][ix];
            if(cell_indices.empty()) continue;

            for(const size_t & idx : cell_indices)
            {
                const Feature::Ptr &ft = getFeatureByIndex(idx);
                const Corner &cr = ft->corner_;

                if(check_level && (cr.level < min_level || cr.level > max_level))
                    continue;

                const float dx = cr.x - x;
                const float dy = cr.y - y;
                const float dist2 = dx*dx + dy*dy;

                if(dist2 > r2)
                    continue;

                indices.push_back(idx);
            }
        }

    return indices;
}

std::vector<size_t> KeyFrame::getFeaturesInGrid(const int r, const int c) const
{
    return grid_[r][c];
}

void KeyFrame::updateConnections()
{
    if(isBad())
        return;

    const std::vector<size_t> indices = getMapPointMatchIndices();
    const std::vector<MapPoint::Ptr> mpts = getMapPoints();

    std::map<KeyFrame::Ptr, int> connection_counter;

    for(const size_t &idx : indices)
    {
        const MapPoint::Ptr &mpt = mpts[idx];

        if(mpt->isBad())
        {
            removeMapPointMatchByIndex(idx);
            continue;
        }

        const std::map<KeyFrame::Ptr, size_t> observations = mpt->getObservations();
        for(const auto &obs : observations)
        {
            if(obs.first->id_ == id_)
                continue;
            connection_counter[obs.first]++;
        }
    }

    if(connection_counter.empty())
    {
        setBad();
        return;
    }

    // TODO how to select proper connections
    int connection_threshold = Config::minConnectionObservations();

    KeyFrame::Ptr best_unfit_keyframe;
    int best_unfit_connections = 0;
    std::vector<std::pair<int, KeyFrame::Ptr> > weight_connections;
    for(const auto &obs : connection_counter)
    {
        if(obs.second < connection_threshold)
        {
            best_unfit_keyframe = obs.first;
            best_unfit_connections = obs.second;
        }
        else
        {
            obs.first->addConnection(shared_from_this(), obs.second);
            weight_connections.emplace_back(std::make_pair(obs.second, obs.first));
        }
    }

    if(weight_connections.empty())
    {
        best_unfit_keyframe->addConnection(shared_from_this(), best_unfit_connections);
        weight_connections.emplace_back(std::make_pair(best_unfit_connections, best_unfit_keyframe));
    }

    //! sort by weight
    std::sort(weight_connections.begin(), weight_connections.end(),
              [](const std::pair<int, KeyFrame::Ptr> &a, const std::pair<int, KeyFrame::Ptr> &b){ return a.first > b.first; });

    //! update
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        connectedKeyFrames_.clear();
        for(const auto &item : weight_connections)
        {
            connectedKeyFrames_.insert(std::make_pair(item.second, item.first));
        }

        orderedConnectedKeyFrames_ =
            std::multimap<int, KeyFrame::Ptr>(weight_connections.begin(), weight_connections.end());
    }
}

std::set<KeyFrame::Ptr> KeyFrame::getConnectedKeyFrames(int num, int min_fts)
{
    std::lock_guard<std::mutex> lock(mutex_connection_);

    std::set<KeyFrame::Ptr> connected_keyframes;
    if(num == -1) num = (int) orderedConnectedKeyFrames_.size();

    int count = 0;
    const auto end = orderedConnectedKeyFrames_.rend();
    for(auto it = orderedConnectedKeyFrames_.rbegin(); it != end && it->first >= min_fts && count < num; it++, count++)
    {
        connected_keyframes.insert(it->second);
    }

    return connected_keyframes;
}

std::set<KeyFrame::Ptr> KeyFrame::getSubConnectedKeyFrames(int num)
{
    std::set<KeyFrame::Ptr> connected_keyframes = getConnectedKeyFrames();

    std::map<KeyFrame::Ptr, int> candidate_keyframes;
    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        std::set<KeyFrame::Ptr> sub_connected_keyframe = kf->getConnectedKeyFrames();
        for(const KeyFrame::Ptr &sub_kf : sub_connected_keyframe)
        {
            if(connected_keyframes.count(sub_kf) || sub_kf == shared_from_this())
                continue;

            if(candidate_keyframes.count(sub_kf))
                candidate_keyframes.find(sub_kf)->second++;
            else
                candidate_keyframes.emplace(sub_kf, 1);
        }
    }

    std::set<KeyFrame::Ptr> sub_connected_keyframes;
    if(num == -1)
    {
        for(const auto &item : candidate_keyframes)
            sub_connected_keyframes.insert(item.first);

        return sub_connected_keyframes;
    }

    //! stort by order
    std::map<int, KeyFrame::Ptr, std::greater<int> > ordered_candidate_keyframes;
    for(const auto &item : candidate_keyframes)
    {
        ordered_candidate_keyframes.emplace(item.second, item.first);
    }

    //! get best (num) keyframes
    for(const auto &item : ordered_candidate_keyframes)
    {
        sub_connected_keyframes.insert(item.second);
        if(sub_connected_keyframes.size() >= num)
            break;
    }

    return sub_connected_keyframes;
}

void KeyFrame::setBad()
{
    if(id_ == 0)
        return;

    std::cout << "The keyframe " << id_ << " was set to be earased." << std::endl;

    std::vector<size_t> indices = getMapPointMatchIndices();
    std::vector<MapPoint::Ptr> mpts = getMapPoints();

    for(const size_t &idx : indices)
    {
        mpts[idx]->removeObservation(shared_from_this());
        removeMapPointMatchByIndex(idx);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        isBad_ = true;

        for(const auto &connect : connectedKeyFrames_)
        {
            connect.first->removeConnection(shared_from_this());
        }

        connectedKeyFrames_.clear();
        orderedConnectedKeyFrames_.clear();
        mpts_matched_.clear();
    }
    // TODO change refKF
}

bool KeyFrame::isBad()
{
    std::lock_guard<std::mutex> lock(mutex_connection_);
    return isBad_;
}

void KeyFrame::addConnection(const KeyFrame::Ptr &kf, const int weight)
{
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);

        if(!connectedKeyFrames_.count(kf))
            connectedKeyFrames_[kf] = weight;
        else if(connectedKeyFrames_[kf] != weight)
            connectedKeyFrames_[kf] = weight;
        else
            return;
    }

    updateOrderedConnections();
}

void KeyFrame::updateOrderedConnections()
{
    std::lock_guard<std::mutex> lock(mutex_connection_);
    orderedConnectedKeyFrames_.clear();
    for(const auto &connect : connectedKeyFrames_)
    {
        auto it = orderedConnectedKeyFrames_.lower_bound(connect.second);
        orderedConnectedKeyFrames_.insert(it, std::pair<int, KeyFrame::Ptr>(connect.second, connect.first));
    }
}

void KeyFrame::removeConnection(const KeyFrame::Ptr &kf)
{
    {
        std::lock_guard<std::mutex> lock(mutex_connection_);
        if(connectedKeyFrames_.count(kf))
        {
            connectedKeyFrames_.erase(kf);
        }
    }

    updateOrderedConnections();
}

void KeyFrame::assignFeaturesToGrid()
{
    int num_reserve = fts_.size() / (GRID_COLS * GRID_ROWS);
    for(int r = 0; r < GRID_ROWS; r++)
        for(int c = 0; c < GRID_COLS; c++)
            grid_[r][c].reserve(num_reserve);

    for(size_t idx = 0; idx < N_; idx++)
    {
        const Corner &cr =  fts_[idx]->corner_;

        int pos_x, pos_y;
        if(getGridPos(cr.x, cr.y, pos_x, pos_y))
            grid_[pos_y][pos_x].push_back(idx);
    }
}

inline bool KeyFrame::getGridPos(const float x, const float y, int &pos_x, int &pos_y)
{
    pos_x = static_cast<int>(x * grid_col_inv_);//! x / cols * GRID_COLS;
    pos_y = static_cast<int>(y * grid_row_inv_);

    if(pos_x < 0 || pos_y < 0 || pos_x >= GRID_COLS || pos_y >= GRID_ROWS)
        return false;
    else
        return true;
}

}