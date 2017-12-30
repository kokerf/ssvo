#include "config.hpp"
#include "system.hpp"
#include "optimizer.hpp"
#include "alignment.hpp"

namespace ssvo{

std::string Config::FileName;

System::System(std::string config_file) :
    stage_(STAGE_FIRST_FRAME), status_(STATUS_INITAL_RESET)
{
    LOG_ASSERT(!config_file.empty()) << "Empty Config file input!!!";
    Config::FileName = config_file;

    const double fps = Config::cameraFps();
    //! image
    const int width = Config::imageWidth();
    const int height = Config::imageHeight();
    const int level = Config::imageTopLevel();
    const int image_border = Align2DI::PatchSize;
    //! camera
    const cv::Mat K = Config::cameraIntrinsic();
    const cv::Mat DistCoef = Config::cameraDistCoef();
    //! corner detector
    const int grid_size = Config::gridSize();
    const int grid_min_size = Config::gridMinSize();
    const int fast_max_threshold = Config::fastMaxThreshold();
    const int fast_min_threshold = Config::fastMinThreshold();

    camera_ = Camera::create(width, height, K, DistCoef);
    fast_detector_ = FastDetector::create(width, height, image_border, level+1, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);
    feature_tracker_ = FeatureTracker::create(width, height, grid_size, true);
    initializer_ = Initializer::create(fast_detector_, true);
    mapper_ = LocalMapper::create(fast_detector_, fps, true, false);
    viewer_ = Viewer::create(mapper_->map_, cv::Size(width, height));

}

void System::process(const cv::Mat &image, const double timestamp)
{
    //! get gray image
    rgb_ = image;
    cv::Mat gray = image.clone();
    if(gray.channels() == 3)
        cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);

    current_frame_ = Frame::create(gray, timestamp, camera_);
    current_frame_->setRefKeyFrame(reference_keyframe_);

    if(STAGE_NORMAL_FRAME == stage_)
    {
        status_ = tracking();
    }
    else if(STAGE_SECOND_FRAME== stage_)
    {
        status_ = processSecondFrame();
    }
    else if(STAGE_FIRST_FRAME == stage_)
    {
        status_ = processFirstFrame();
    }
    else if(STAGE_RELOCALIZING == stage_)
    {

    }

    finishFrame();
}

System::Status System::processFirstFrame()
{
    InitResult result = initializer_->addImage(current_frame_);
    if(result == RESET)
        return STATUS_INITAL_RESET;

    return STATUS_INITAL_SUCCEED;
}

System::Status System::processSecondFrame()
{
    InitResult result = initializer_->addImage(current_frame_);

    if(result == RESET)
        return STATUS_INITAL_RESET;
    else if(result == FAILURE)
        return STATUS_INITAL_FALIURE;

    std::vector<Vector3d> points;
    initializer_->createInitalMap(points, Config::mapScale());
    mapper_->createInitalMap(initializer_->getReferenceFrame(), current_frame_, points);

    LOG(WARNING) << "[System] Start two-view BA";

    KeyFrame::Ptr kf0 = mapper_->map_->getKeyFrame(0);
    KeyFrame::Ptr kf1 = mapper_->map_->getKeyFrame(1);

    LOG_ASSERT(kf0 != nullptr && kf1 != nullptr) << "Can not find intial keyframes in map!";

    Optimizer::twoViewBundleAdjustment(kf0, kf1, true);

    LOG(WARNING) << "[System] End of two-view BA";

    reference_keyframe_ = kf1;
    current_frame_->setPose(reference_keyframe_->pose());
    current_frame_->setRefKeyFrame(reference_keyframe_);

    return STATUS_INITAL_SUCCEED;
}

System::Status System::tracking()
{
    // TODO 先验信息怎么设置？
    current_frame_->setPose(last_frame_->pose());
    //! alignment by SE3
    AlignSE3 align;
    align.run(last_frame_, current_frame_, Config::alignTopLevel(), 30, 1e-8);

    //! track local map
    LOG(WARNING) << "[System] Tracking local map";
    int matches = feature_tracker_->reprojectLoaclMap(current_frame_);
    LOG(WARNING) << "[System] Track with " << matches << " points";

    // TODO tracking status
    if(matches < Config::minQualityFts())
        return STATUS_TRACKING_BAD;

    //! motion-only BA
    LOG(WARNING) << "[System] Motion-Only BA";
    Optimizer::motionOnlyBundleAdjustment(current_frame_, true);
    LOG(WARNING) << "[System] Finish Motion-Only BA";

    return STATUS_TRACKING_GOOD;
}

void System::finishFrame()
{
    Stage last_stage = stage_;
    if(STAGE_NORMAL_FRAME == stage_)
    {
        if(STATUS_TRACKING_GOOD)
        {
            changeReferenceKeyFrame();
        }
    }
    else if(STAGE_SECOND_FRAME == stage_)
    {
        switch(status_)
        {
            case STATUS_INITAL_FALIURE : stage_ = STAGE_SECOND_FRAME; break;
            case STATUS_INITAL_RESET   : stage_ = STAGE_FIRST_FRAME; break;
            case STATUS_INITAL_SUCCEED : stage_ = STAGE_NORMAL_FRAME; break;
            default: break;
        }
    }
    else if(STAGE_FIRST_FRAME == stage_)
    {
        if(STATUS_INITAL_SUCCEED == status_)
            stage_ = STAGE_SECOND_FRAME;
    }

    //! update
    last_frame_ = current_frame_;

    LOG(WARNING) << "[System] Finsh Current Frame with Stage: " << stage_;

    //! display
    viewer_->showImage(rgb_);
    viewer_->setCurrentCameraPose(current_frame_->pose().matrix());
    showImage(last_stage);
}

bool System::changeReferenceKeyFrame()
{
    std::map<KeyFrame::Ptr, int> overlap_kf = current_frame_->getOverLapKeyFrames();
    const int overlap = overlap_kf[reference_keyframe_];

    double median_depth = std::numeric_limits<double>::max();
    double min_depth = std::numeric_limits<double>::max();
    current_frame_->getSceneDepth(median_depth, min_depth);

    SE3d T_cur_from_ref = current_frame_->Tcw() * reference_keyframe_->pose();
    Vector3d tran = T_cur_from_ref.translation();

    bool c1 = tran.dot(tran) > 0.12 * median_depth;
    bool c2 = static_cast<double>(overlap) / reference_keyframe_->N() < 0.8;

    //! create new keyFrame
    if(c1 || c2)
    {
        KeyFrame::Ptr new_keyframe = KeyFrame::create(current_frame_);
//        mapper_->insertNewFrame(current_frame_, new_keyframe, median_depth, min_depth);
    }
    //! change reference keyframe
    else
    {
        int max_overlap = -1;
        KeyFrame::Ptr best_keyframe;
        for(const auto &op : overlap_kf)
        {
            if(op.second <= max_overlap)
                continue;

            max_overlap = op.second;
            best_keyframe = op.first;
        }

        if(max_overlap < 1.2 * overlap)
            return false;

        reference_keyframe_ = best_keyframe;
//        mapper_->insertNewFrame(current_frame_, nullptr, median_depth, min_depth);
    }

    return true;
}

void System::showImage(Stage stage)
{
    cv::Mat image = rgb_;
    if(image.channels() < 3)
        cv::cvtColor(image, image, CV_GRAY2RGB);

    if(STAGE_NORMAL_FRAME == stage)
    {
        std::vector<Feature::Ptr> fts = current_frame_->getFeatures();
        for(const Feature::Ptr &ft : fts)
        {
            Vector2d px = ft->px;
            cv::circle(image, cv::Point2d(px[0], px[1]), 2, cv::Scalar(0, 255, 0), -1);
        }
    }
    else if(STAGE_SECOND_FRAME == stage)
    {
        initializer_->drowOpticalFlow(image, image);
    }

    cv::imshow("SSVO Current Image", image);
    cv::waitKey(1);
}

}

