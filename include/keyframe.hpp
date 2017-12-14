#ifndef _KEYFRAME_HPP_
#define _KEYFRAME_HPP_

#include "global.hpp"
#include "frame.hpp"

namespace ssvo
{

class Map;

class KeyFrame: public Frame, public std::enable_shared_from_this<KeyFrame>
{
public:

    typedef std::shared_ptr<KeyFrame> Ptr;

    void updateConnections();

    MapPoints getMapPoints();

    std::vector<KeyFrame::Ptr> getConnectedKeyFrames();

    inline static KeyFrame::Ptr create(const Frame::Ptr frame)
    { return std::make_shared<KeyFrame>(KeyFrame(frame)); }

private:
    KeyFrame(const Frame::Ptr frame);

    void addConnection(KeyFrame::Ptr kf, const int& weight);

    void updateOrderedConnections();

public:

    static uint64_t next_id_;

    const uint64_t frame_id_;

    std::map<KeyFrame::Ptr, int> connectedKeyFrames_;

    std::multimap<int, KeyFrame::Ptr> orderedConnectedKeyFrames_;

};

}

#endif
