#ifndef _SSVO_SYSTEM_HPP_
#define _SSVO_SYSTEM_HPP_

#include "global.hpp"
#include "frame.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "initializer.hpp"
#include "feature_detector.hpp"
#include "feature_tracker.hpp"
#include "viewer.hpp"

namespace ssvo {

class System: public noncopyable
{
public:
    enum Stage{
        STAGE_INITAL_RESET,
        STAGE_INITAL_PROCESS,
        STAGE_TRACKING,
        STAGE_RELOCALIZING
    };

    System(std::string config_file);

    void process(const cv::Mat& image, const double timestamp);

private:

    void processFrame();
    Stage tracking();
    Stage processFirstFrame();
    Stage processSecondFrame();

private:

    Stage stage_;

    Camera::Ptr camera_;
    Map::Ptr map_;
    FastDetector::Ptr fast_detector_;
    FeatureTracker::Ptr feature_tracker_;
    Initializer::Ptr initializer_;

    Viewer::Ptr viewer_;

    std::thread viewer_thread_;

    cv::Mat image_;
    Frame::Ptr current_frame_;
    Frame::Ptr last_frame_;
    KeyFrame::Ptr reference_keyframe_;
};

}// namespce ssvo

#endif //SSVO_SYSTEM_HPP
