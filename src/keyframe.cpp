#include "keyframe.hpp"

namespace ssvo{

uint64_t KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(const Frame::Ptr frame):
    Frame(frame->img_pyr_, next_id_++, frame->timestamp_, frame->cam_), frame_id_(frame->id_)
{
    fts_ = frame->fts_;
    mpts_ = frame->mpts_;
    setPose(frame->pose());
}

void KeyFrame::updateObservation()
{
    Features::iterator fts_iter = fts_.begin();
    Features::iterator fts_end = fts_.end();

    for(;fts_iter != fts_end; fts_iter++)
    {
        Feature::Ptr &ft = (*fts_iter);
        if(ft->mpt != nullptr)
        {
            ft->mpt->addObservation((KeyFrame::Ptr)this, ft);
        }
    }
}

}