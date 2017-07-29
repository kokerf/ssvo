#ifndef _KEYFRAME_HPP_
#define _KEYFRAME_HPP_

#include "frame.hpp"
#include "global.hpp"

namespace ssvo
{

class KeyFrame: public Frame
{
public:

    typedef std::shared_ptr<KeyFrame> Ptr;

    KeyFrame(const Frame::Ptr frame);

    void updateObservation();

    inline static KeyFrame::Ptr create(const Frame::Ptr frame) {return KeyFrame::Ptr(new KeyFrame(frame));}

private:

    static uint64_t next_id_;

    uint64_t frame_id_;
};

}

#endif
