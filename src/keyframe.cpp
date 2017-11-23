#include "config.hpp"
#include "map.hpp"
#include "keyframe.hpp"

namespace ssvo{

uint64_t KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(const Frame::Ptr frame):
    Frame(frame->img_pyr_, next_id_++, frame->timestamp_, frame->cam_), frame_id_(frame->id_)
{
    fts_ = frame->getFeatures();
    setPose(frame->pose());
}

void KeyFrame::updateConnections()
{
    std::map<KeyFrame::Ptr, int> connection_counter;

    for(const Feature::Ptr ft : fts_)
    {
        if(ft->mpt == nullptr)
            continue;

        std::map<KeyFrame::Ptr, Feature::Ptr> observations = ft->mpt->getObservations();
        for(auto obs : observations)
        {
            if(obs.first->id_ == id_)
                continue;
            connection_counter[obs.first]++;
        }
    }

    LOG_ASSERT(!connection_counter.empty()) << "No connections find in KF: " << id_;

    // TODO how to select proper connections
    int connection_threshold = Config::minConnectionObservations();

    KeyFrame::Ptr best_unfit_keyframe;
    int best_unfit_connections = 0;
    std::vector<std::pair<int, KeyFrame::Ptr> > weight_connections;
    for(const auto obs : connection_counter)
    {
        if(obs.second < connection_threshold)
        {
            best_unfit_keyframe = obs.first;
            best_unfit_connections = obs.second;
        }
        else
        {
            obs.first->addConnection(shared_from_this(), obs.second);
            weight_connections.push_back(std::make_pair(obs.second, obs.first));
        }
    }

    if(weight_connections.empty())
    {
        best_unfit_keyframe->addConnection(shared_from_this(), best_unfit_connections);
        weight_connections.push_back(std::make_pair(best_unfit_connections, best_unfit_keyframe));
    }

    //! sort by weight
    std::sort(weight_connections.begin(), weight_connections.end(),
              [](const std::pair<int, KeyFrame::Ptr> &a, const std::pair<int, KeyFrame::Ptr> &b){ return a.first > b.first; });

    //! update
    connectedKeyFrames_.clear();
    for(const auto it: weight_connections)
    {
        connectedKeyFrames_.insert(std::make_pair(it.second, it.first));
    }

    orderedConnectedKeyFrames_ = std::multimap<int, KeyFrame::Ptr>(weight_connections.begin(), weight_connections.end());
}

void KeyFrame::addConnection(KeyFrame::Ptr kf, const int &weight)
{
    if(!connectedKeyFrames_.count(kf))
        connectedKeyFrames_[kf] = weight;
    else if(connectedKeyFrames_[kf] != weight)
        connectedKeyFrames_[kf] = weight;
    else
        return;

    updateOrderedConnections();
}

void KeyFrame::updateOrderedConnections()
{
    orderedConnectedKeyFrames_.clear();
    for(std::pair<KeyFrame::Ptr, int> connect : connectedKeyFrames_)
    {
        std::multimap<int, KeyFrame::Ptr>::iterator it = orderedConnectedKeyFrames_.lower_bound(connect.second);
        orderedConnectedKeyFrames_.insert(it, std::pair<int, KeyFrame::Ptr>(connect.second, connect.first));
    }
}

MapPoints KeyFrame::getMapPoints()
{
    MapPoints mpts;
    mpts.reserve(fts_.size());
    for(const Feature::Ptr ft : fts_)
    {
        if(ft->mpt != nullptr)
            mpts.push_back(ft->mpt);
    }

    return mpts;
}

}