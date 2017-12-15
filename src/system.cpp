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

    const int fps = Config::cameraFps();
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

    map_ = Map::create();
    camera_ = Camera::create(width, height, K, DistCoef);
    fast_detector_ = FastDetector::create(width, height, image_border, level+1, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);
    feature_tracker_ = FeatureTracker::create(width, height, grid_size);
    initializer_ = Initializer::create(fast_detector_);
    viewer_ = Viewer::create(map_, cv::Size(width, height));

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
    InitResult result = initializer_->addFirstImage(current_frame_);
    if(result == RESET)
        return STATUS_INITAL_RESET;

    return STATUS_INITAL_SUCCEED;
}

System::Status System::processSecondFrame()
{
    InitResult result = initializer_->addSecondImage(current_frame_);

    if(result == RESET)
        return STATUS_INITAL_RESET;
    else if(result == FAILURE)
        return STATUS_INITAL_FALIURE;

    map_->clear();
    initializer_->createInitalMap(map_, Config::mapScale());

    LOG(WARNING) << "Start two-view BA";

    ssvo::Optimizer optimizer;
    std::vector<KeyFrame::Ptr> kfs = map_->getAllKeyFramesOrderedByID();
    LOG_ASSERT(kfs.size() == 2) << "Error number of keyframes in map after initailizer: " << kfs.size();
    LOG_ASSERT(kfs[0]->id_ == 0 && kfs[1]->id_ == 1) << "Error id of keyframe: " << kfs[0]->id_ << ", " << kfs[0]->id_;

    optimizer.twoViewBundleAdjustment(kfs[0], kfs[1], nullptr);
    optimizer.report(true);

    LOG(WARNING) << "End of two-view BA";

    reference_keyframe_ = kfs[1];
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
    LOG(INFO) << "Tracking local map";
    int matches = feature_tracker_->reprojectLoaclMap(current_frame_, map_);
    LOG(INFO) << "Track " << matches << "points";

    // TODO tracking status
    if(matches < Config::minQualityFts())
        return STATUS_TRACKING_BAD;

    return STATUS_TRACKING_GOOD;
}

void System::finishFrame()
{
    Stage last_stage = stage_;
    if(STAGE_NORMAL_FRAME == stage_)
    {
        if(STATUS_TRACKING_BAD)
        {

        }
        else if(STATUS_TRACKING_INSUFFICIENT)
        {

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


    // TODO keyframe selection

    //! update
    last_frame_ = current_frame_;

    LOG(WARNING) << "System Stage: " << stage_;

    //! display
    viewer_->showImage(rgb_);
    viewer_->setCurrentCameraPose(current_frame_->pose().matrix());
    showImage(last_stage);
}

void System::showImage(Stage stage)
{
    cv::Mat image = rgb_;
    if(image.channels() < 3)
        cv::cvtColor(image, image, CV_GRAY2RGB);

    if(STAGE_NORMAL_FRAME == stage)
    {
        std::vector<Feature::Ptr> fts = current_frame_->getFeatures();
        for (Feature::Ptr ft : fts)
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

