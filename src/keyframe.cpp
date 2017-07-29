#include "keyframe.hpp"

namespace ssvo{

uint64_t KeyFrame::next_id_ = 0;

KeyFrame::KeyFrame(const Frame::Ptr frame)
{
    id_ = next_id_++;
    frame_id_ = frame->id();
    timestamp_ = frame->timeStamp();
    cam_ = frame->cam_;
    img_pyr_ = frame->img_pyr_;
    fts_ = frame->fts_;
    mpts_ = frame->mpts_;
    q_ = frame->getRotation();
    t_ = frame->getTranslation();
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