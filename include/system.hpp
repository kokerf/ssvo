#ifndef _SSVO_SYSTEM_HPP_
#define _SSVO_SYSTEM_HPP_

#include "global.hpp"
#include "frame.hpp"
#include "keyframe.hpp"
#include "map.hpp"
#include "initializer.hpp"
#include "feature_detector.hpp"
#include "feature_tracker.hpp"
#include "local_mapping.hpp"
#include "viewer.hpp"

namespace ssvo {

class System: public noncopyable
{
public:
    enum Stage{
        STAGE_FIRST_FRAME,
        STAGE_SECOND_FRAME,
        STAGE_NORMAL_FRAME,
        STAGE_RELOCALIZING
    };

    enum Status {
        STATUS_INITAL_RESET,
        STATUS_INITAL_FALIURE,
        STATUS_INITAL_SUCCEED,
        STATUS_TRACKING_BAD,
        STATUS_TRACKING_GOOD,
    };

    System(std::string config_file);

    void process(const cv::Mat& image, const double timestamp);

private:

    void processFrame();

    Status tracking();

    Status processFirstFrame();

    Status processSecondFrame();

    void finishFrame();

    void showImage(Stage stage);

    bool changeReferenceKeyFrame();

private:

    Stage stage_;
    Status status_;

    Camera::Ptr camera_;
    FastDetector::Ptr fast_detector_;
    FeatureTracker::Ptr feature_tracker_;
    Initializer::Ptr initializer_;
    LocalMapper::Ptr mapper_;

    Viewer::Ptr viewer_;

    std::thread viewer_thread_;

    cv::Mat rgb_;
    Frame::Ptr current_frame_;
    Frame::Ptr last_frame_;
    KeyFrame::Ptr reference_keyframe_;
};

}// namespce ssvo

#endif //SSVO_SYSTEM_HPP
