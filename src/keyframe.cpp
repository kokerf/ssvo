#include "config.hpp"
#include "map.hpp"
#include "keyframe.hpp"

namespace ssvo{

uint64_t KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(const Frame::Ptr frame):
    Frame(frame->images(), next_id_++, frame->timestamp_, frame->cam_), frame_id_(frame->id_)
{
    mpt_fts_ = frame->features();
    setRefKeyFrame(frame->getRefKeyFrame());
    setPose(frame->pose());
}
void KeyFrame::updateConnections()
{
    Features fts;
    {
        std::lock_guard<std::mutex> lock(mutex_feature_);
        for(const auto &it : mpt_fts_)
            fts.push_back(it.second);
    }

    std::map<KeyFrame::Ptr, int> connection_counter;

    for(const Feature::Ptr &ft : fts)
    {
        const MapPoint::Ptr &mpt = ft->mpt_;

        if(mpt->isBad())
        {
            removeFeature(ft);
            continue;
        }

        const std::map<KeyFrame::Ptr, Feature::Ptr> observations = mpt->getObservations();
        for(const auto &obs : observations)
        {
            if(obs.first->id_ == id_)
                continue;
            connection_counter[obs.first]++;
        }
    }

    LOG_ASSERT(!connection_counter.empty()) << " No connections find in KF: " << id_;

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

std::set<KeyFrame::Ptr> KeyFrame::getConnectedKeyFrames(int num)
{
    std::lock_guard<std::mutex> lock(mutex_connection_);

    std::set<KeyFrame::Ptr> kfs;
    if(num == -1) num = (int) orderedConnectedKeyFrames_.size();
    int count = 0;
    for(const auto &ordered_keyframe : orderedConnectedKeyFrames_)
    {
        kfs.insert(ordered_keyframe.second);
        if(++count >= num)
            break;
    }

    return kfs;
}

void KeyFrame::setBad()
{
    if(id_ == 0)
        return;

    {
        std::lock_guard<std::mutex> lock(mutex_connection_);
        for(const auto &connect : connectedKeyFrames_)
        {
            connect.first->removeConnection(shared_from_this());
        }

        connectedKeyFrames_.clear();
        orderedConnectedKeyFrames_.clear();
    }

    std::lock_guard<std::mutex> lock(mutex_feature_);
    for(const auto &it : mpt_fts_)
    {
        it.first->removeObservation(shared_from_this());
    }

    // TODO change refKF
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
    std::lock_guard<std::mutex> lock(mutex_connection_);
    if(connectedKeyFrames_.count(kf))
    {
        connectedKeyFrames_.erase(kf);
        updateOrderedConnections();
    }
}

}