#include "config.hpp"
#include "system.hpp"
#include "optimizer.hpp"
#include "alignment.hpp"

namespace ssvo{

std::string Config::FileName;

System::System(std::string config_file)
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
    viewer_ = Viewer::create(map_);

    stage_ = STAGE_INITAL_RESET;

}

void System::process(const cv::Mat &image, const double timestamp)
{
    //! get gray image
    if(image.channels() == 3)
        cv::cvtColor(image, image_, cv::COLOR_RGB2GRAY);
    else
        image_ = image.clone();

    current_frame_ = Frame::create(image_, timestamp, camera_);
    current_frame_->setRefKeyFrame(reference_keyframe_);

    if(STAGE_TRACKING == stage_)
    {
        tracking();
        cv::waitKey(0);
    }
    else if(STAGE_INITAL_PROCESS== stage_)
    {
        stage_ = processSecondFrame();
    }
    else if(STAGE_INITAL_RESET == stage_)
    {
        stage_ = processFirstFrame();
    }
    else if(STAGE_RELOCALIZING == stage_)
    {

    }

    viewer_->showImage(image);
    viewer_->setCurrentCameraPose(current_frame_->pose().matrix());
    last_frame_ = current_frame_;

    LOG(WARNING) << "System Stage: " << stage_;
}

System::Stage System::processFirstFrame()
{
    InitResult result = initializer_->addFirstImage(current_frame_);
    if(result == RESET)
        return STAGE_INITAL_RESET;

    return STAGE_INITAL_PROCESS;
}

System::Stage System::processSecondFrame()
{
    InitResult result = initializer_->addSecondImage(current_frame_);
    cv::Mat klt_img;
    initializer_->drowOpticalFlow(image_, klt_img);
    cv::imshow("KLTracking", klt_img);

    if(result == RESET)
        return STAGE_INITAL_RESET;
    else if(result == FAILURE)
        return STAGE_INITAL_PROCESS;

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

    return STAGE_TRACKING;
}

System::Stage System::tracking()
{
    // TODO 先验信息怎么设置？
    current_frame_->setPose(last_frame_->pose());
    //! alignment by SE3
    AlignSE3 align;
    align.run(last_frame_, current_frame_, Config::alignTopLevel(), 30, 1e-8);

    Sophus::SE3d T_ref_from_cur = last_frame_->Tcw() * current_frame_->pose();
    AngleAxisd aa; aa = T_ref_from_cur.rotationMatrix();
    LOG(INFO) << "pose_ref: [" << last_frame_->pose().log().transpose();
    LOG(INFO) << "pose_cur: [" << current_frame_->pose().log().transpose();
    LOG(INFO) << "trans: [" << T_ref_from_cur.translation().transpose()
              << "] angle: [" << aa.axis().transpose() << "] * " << aa.angle();

    feature_tracker_->reprojectLoaclMap(current_frame_, map_);
}

}

