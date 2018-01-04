#include "config.hpp"
#include "system.hpp"
#include "optimizer.hpp"
#include "alignment.hpp"

namespace ssvo{

std::string Config::FileName;

System::System(std::string config_file) :
    stage_(STAGE_INITALIZE), status_(STATUS_INITAL_RESET)
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
    else if(STAGE_INITALIZE == stage_)
    {
        status_ = initialize();
    }
    else if(STAGE_RELOCALIZING == stage_)
    {

    }

    finishFrame();
}

System::Status System::initialize()
{
    const Initializer::Result result = initializer_->addImage(current_frame_);

    if(result == Initializer::RESET)
        return STATUS_INITAL_RESET;
    else if(result == Initializer::FAILURE || result == Initializer::READY)
        return STATUS_INITAL_PROCESS;

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
    //! track seeds
    double t0 = (double)cv::getTickCount();
    mapper_->insertNewFrame(current_frame_);
    double t1 = (double)cv::getTickCount();

    // TODO 先验信息怎么设置？
    current_frame_->setPose(last_frame_->pose());
    //! alignment by SE3
    AlignSE3 align;
    align.run(last_frame_, current_frame_, Config::alignTopLevel(), 30, 1e-8);

    //! track local map
    double t2 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Tracking local map";
    int matches = feature_tracker_->reprojectLoaclMap(current_frame_);
    LOG(WARNING) << "[System] Track with " << matches << " points";

    // TODO tracking status
//    if(matches < Config::minQualityFts())
//        return STATUS_TRACKING_BAD;

    //! motion-only BA
    double t3 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Motion-Only BA";
    Optimizer::motionOnlyBundleAdjustment(current_frame_, true);
    LOG(WARNING) << "[System] Finish Motion-Only BA";
    double t4 = (double)cv::getTickCount();

    mapper_->finishFrame();
    double t5 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Time: " << (t1-t0)/cv::getTickFrequency() << " "
                                      << (t2-t1)/cv::getTickFrequency() << " "
                                      << (t3-t2)/cv::getTickFrequency() << " "
                                      << (t4-t3)/cv::getTickFrequency() << " "
                                      << (t5-t4)/cv::getTickFrequency();

    return STATUS_TRACKING_GOOD;
}

void System::finishFrame()
{
    Stage last_stage = stage_;
    if(STAGE_NORMAL_FRAME == stage_)
    {
        if(STATUS_TRACKING_GOOD == status_)
        {
            changeReferenceKeyFrame();
        }
    }
    else if(STAGE_INITALIZE == stage_)
    {
        if(STATUS_INITAL_SUCCEED == status_)
            stage_ = STAGE_NORMAL_FRAME;
        else if(STATUS_INITAL_RESET == status_)
            initializer_->reset();
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
    std::map<KeyFrame::Ptr, int> overlap_kfs = current_frame_->getOverLapKeyFrames();
    const int overlap = overlap_kfs[reference_keyframe_];

    double median_depth = std::numeric_limits<double>::max();
    double min_depth = std::numeric_limits<double>::max();
    current_frame_->getSceneDepth(median_depth, min_depth);

    bool c1 = true;
    for(const auto &op_kf : overlap_kfs)
    {
        SE3d T_cur_from_ref = current_frame_->Tcw() * op_kf.first->pose();
        Vector3d tran = T_cur_from_ref.translation();
        double dist1 = tran.dot(tran);
        double dist2 = 0.2 * (T_cur_from_ref.rotationMatrix() - Matrix3d::Identity()).norm();
        if(dist1 + dist2 < 0.12 * median_depth)
        {
            c1 = false;
            break;
        }
    }

    bool c2 = static_cast<double>(overlap) / reference_keyframe_->N() < 0.5;

    //! create new keyFrame
    if(c1 || c2)
    {
        KeyFrame::Ptr new_keyframe = KeyFrame::create(current_frame_);
        mapper_->insertNewKeyFrame(new_keyframe, median_depth, min_depth);

        reference_keyframe_ = new_keyframe;
    }
    //! change reference keyframe
    else
    {
        int max_overlap = -1;
        KeyFrame::Ptr best_keyframe;
        for(const auto &op_kf : overlap_kfs)
        {
            if(op_kf.second <= max_overlap)
                continue;

            max_overlap = op_kf.second;
            best_keyframe = op_kf.first;
        }

        if(max_overlap < 1.2 * overlap)
            return false;

        reference_keyframe_ = best_keyframe;
    }

    return true;
}

void System::showImage(Stage stage)
{
//    cv::Mat image = rgb_;
//    if(image.channels() < 3)
//        cv::cvtColor(image, image, CV_GRAY2RGB);

    cv::Mat image;
    if(STAGE_NORMAL_FRAME == stage)
    {
        mapper_->drowTrackedPoints(image);
    }
    else if(STAGE_INITALIZE == stage)
    {
        initializer_->drowOpticalFlow(image);
    }

    cv::imshow("SSVO Current Image", image);
    cv::waitKey(1);
}

}

