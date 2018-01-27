#include "config.hpp"
#include "system.hpp"
#include "optimizer.hpp"
#include "image_alignment.hpp"
#include "feature_alignment.hpp"

namespace ssvo{

std::string Config::FileName;

System::System(std::string config_file) :
    stage_(STAGE_INITALIZE), status_(STATUS_INITAL_RESET),
    last_frame_(nullptr), current_frame_(nullptr), reference_keyframe_(nullptr)
{
    LOG_ASSERT(!config_file.empty()) << "Empty Config file input!!!";
    Config::FileName = config_file;

    double fps = Config::cameraFps();
    if(fps < 1.0) fps = 1.0;
    //! image
    const int width = Config::imageWidth();
    const int height = Config::imageHeight();
    const int level = Config::imageTopLevel();
    const int image_border = AlignPatch::Size;
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
    feature_tracker_ = FeatureTracker::create(width, height, grid_size, image_border, true);
    initializer_ = Initializer::create(fast_detector_, true);
    mapper_ = LocalMapper::create(fps, true, false);
    DepthFilter::Callback depth_fliter_callback = std::bind(&LocalMapper::createFeatureFromSeed, mapper_, std::placeholders::_1);
    depth_filter_ = DepthFilter::create(fast_detector_, depth_fliter_callback, true);
    viewer_ = Viewer::create(mapper_->map_, cv::Size(width, height));

    mapper_->startMainThread();
    //    depth_filter_->startMainThread();

    time_ = 1000.0/fps;
}

System::~System()
{
    viewer_->setStop();
    depth_filter_->stopMainThread();
    mapper_->stopMainThread();

    viewer_->waitForFinish();
}

void System::process(const cv::Mat &image, const double timestamp)
{
    //! get gray image
    double t0 = (double)cv::getTickCount();
    rgb_ = image;
    cv::Mat gray = image.clone();
    if(gray.channels() == 3)
        cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);

    current_frame_ = Frame::create(gray, timestamp, camera_);
    current_frame_->setRefKeyFrame(reference_keyframe_);
    double t1 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Frame " << current_frame_->id_ << " create time: " << (t1-t0)/cv::getTickFrequency();

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
    initializer_->createInitalMap(Config::mapScale());
    mapper_->createInitalMap(initializer_->getReferenceFrame(), current_frame_);

    LOG(WARNING) << "[System] Start two-view BA";

    KeyFrame::Ptr kf0 = mapper_->map_->getKeyFrame(0);
    KeyFrame::Ptr kf1 = mapper_->map_->getKeyFrame(1);

    LOG_ASSERT(kf0 != nullptr && kf1 != nullptr) << "Can not find intial keyframes in map!";

    Optimizer::twoViewBundleAdjustment(kf0, kf1, true);

    LOG(WARNING) << "[System] End of two-view BA";

    current_frame_->setPose(kf1->pose());
    current_frame_->setRefKeyFrame(kf1);
    reference_keyframe_ = kf1;
    last_keyframe_ = kf1;

    depth_filter_->createSeeds(kf0);
    depth_filter_->createSeeds(kf1, current_frame_);

    initializer_->reset();

    return STATUS_INITAL_SUCCEED;
}

System::Status System::tracking()
{
    //! track seeds
    double t0 = (double)cv::getTickCount();
    depth_filter_->trackFrame(last_frame_, current_frame_);
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
    mapper_->refineMapPoints(20);
    LOG(WARNING) << "[System] Finish Motion-Only BA";
    double t4 = (double)cv::getTickCount();

    depth_filter_->insertFrame(current_frame_);
    if(createNewKeyFrame())
    {
        mapper_->insertKeyFrame(reference_keyframe_);
        int new_seeds = depth_filter_->createSeeds(reference_keyframe_);

        LOG(INFO) << "[System] New created depth filter seeds: " << new_seeds;
    }

    double t5 = (double)cv::getTickCount();
    calcLightAffine();

    double t6 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Time: " << (t1-t0)/cv::getTickFrequency() << " "
                                      << (t2-t1)/cv::getTickFrequency() << " "
                                      << (t3-t2)/cv::getTickFrequency() << " "
                                      << (t4-t3)/cv::getTickFrequency() << " "
                                      << (t5-t4)/cv::getTickFrequency() << " "
                                      << (t6-t5)/cv::getTickFrequency()
                 << ", Total: " << (t6-t0)/cv::getTickFrequency();

    //！ save frame pose
    frame_timestamp_buffer_.push_back(current_frame_->timestamp_);
    reference_keyframe_buffer_.push_back(current_frame_->getRefKeyFrame());
    frame_pose_buffer_.push_back(current_frame_->getRefKeyFrame()->Tcw() * current_frame_->pose());

    return STATUS_TRACKING_GOOD;
}

void System::calcLightAffine()
{
    std::vector<Feature::Ptr> fts_last;
    last_frame_->getFeatures(fts_last);

    const cv::Mat img_last = last_frame_->getImage(0);
    const cv::Mat img_curr = current_frame_->getImage(0).clone() * 1.3;

    const int size = 4;
    const int patch_area = size*size;
    const int N = (int)fts_last.size();
    cv::Mat patch_buffer_last = cv::Mat::zeros(N, patch_area, CV_32FC1);
    cv::Mat patch_buffer_curr = cv::Mat::zeros(N, patch_area, CV_32FC1);

    int count = 0;
    for(int i = 0; i < N; ++i)
    {
        const Feature::Ptr ft_last = fts_last[i];
        const Feature::Ptr ft_curr = current_frame_->getFeatureByMapPoint(ft_last->mpt_);

        if(ft_curr == nullptr)
            continue;

        utils::interpolateMat<uchar, float, size>(img_last, patch_buffer_last.ptr<float>(count), ft_last->px_[0], ft_last->px_[1]);
        utils::interpolateMat<uchar, float, size>(img_curr, patch_buffer_curr.ptr<float>(count), ft_curr->px_[0], ft_curr->px_[1]);

        count++;
    }

    patch_buffer_last.resize(count);
    patch_buffer_curr.resize(count);

    if(count < 20)
    {
        Frame::light_affine_a_ = 1;
        Frame::light_affine_b_ = 0;
        return;
    }

    float a=1;
    float b=0;
    calculateLightAffine(patch_buffer_last, patch_buffer_curr, a, b);
    Frame::light_affine_a_ = a;
    Frame::light_affine_b_ = b;

//    std::cout << "a: " << a << " b: " << b << std::endl;
}

bool System::createNewKeyFrame()
{
    std::map<KeyFrame::Ptr, int> overlap_kfs = current_frame_->getOverLapKeyFrames();

    std::vector<Feature::Ptr> fts;
    current_frame_->getFeatures(fts);
    std::map<MapPoint::Ptr, Feature::Ptr> mpt_ft;
    for(const Feature::Ptr &ft : fts)
    {
        mpt_ft.emplace(ft->mpt_, ft);
    }

    KeyFrame::Ptr max_overlap_keyframe;
    int max_overlap = 0;
    for(const auto &olp_kf : overlap_kfs)
    {
        if(olp_kf.second < max_overlap || (olp_kf.second == max_overlap && olp_kf.first->id_ < max_overlap_keyframe->id_))
            continue;

        max_overlap_keyframe = olp_kf.first;
        max_overlap = olp_kf.second;
    }

    //! check distance
    bool c1 = true;
    double median_depth = std::numeric_limits<double>::max();
    double min_depth = std::numeric_limits<double>::max();
    current_frame_->getSceneDepth(median_depth, min_depth);
//    for(const auto &ovlp_kf : overlap_kfs)
//    {
//        SE3d T_cur_from_ref = current_frame_->Tcw() * ovlp_kf.first->pose();
//        Vector3d tran = T_cur_from_ref.translation();
//        double dist1 = tran.dot(tran);
//        double dist2 = 0.1 * (T_cur_from_ref.rotationMatrix() - Matrix3d::Identity()).norm();
//        double dist = dist1 + dist2;
////        std::cout << "d1: " << dist1 << ". d2: " << dist2 << std::endl;
//        if(dist  < 0.10 * median_depth)
//        {
//            c1 = false;
//            break;
//        }
//    }

    SE3d T_cur_from_ref = current_frame_->Tcw() * last_keyframe_->pose();
    Vector3d tran = T_cur_from_ref.translation();
    double dist1 = tran.dot(tran);
    double dist2 = 0.01 * (T_cur_from_ref.rotationMatrix() - Matrix3d::Identity()).norm();
    if(dist1+dist2  < 0.01 * median_depth)
        c1 = false;

    //! check disparity
    std::list<float> disparities;
    const int threahold = (int)max_overlap * 0.6;
    for(const auto &ovlp_kf : overlap_kfs)
    {
        if(ovlp_kf.second < threahold)
            continue;

        std::vector<float> disparity;
        disparity.reserve(ovlp_kf.second);
        MapPoints mpts;
        ovlp_kf.first->getMapPoints(mpts);
        for(const MapPoint::Ptr &mpt : mpts)
        {
            Feature::Ptr ft_ref = mpt->findObservation(ovlp_kf.first);
            Feature::Ptr ft_cur = mpt_ft[mpt];
            if(ft_ref != nullptr && ft_cur != nullptr)
            {
                const Vector2d px(ft_ref->px_ - ft_cur->px_);
                disparity.push_back(px.norm());
            }
        }

        std::sort(disparity.begin(), disparity.end());
        float disp = disparity.at(disparity.size()/2);
        disparities.push_back(disp);
    }
    disparities.sort();

    LOG(INFO) << "[System] Max overlap: " << max_overlap << " min disaprity " << disparities.front() << ", average ";

//    int all_features = current_frame_->featureNumber() + current_frame_->seedNumber();
    bool c2 = disparities.front() > 100;//options_.min_disparity;
    bool c3 = current_frame_->featureNumber() < reference_keyframe_->featureNumber() * 0.7;
//    bool c4 = current_frame_->featureNumber() < reference_keyframe_->featureNumber() * 0.9;

    //! create new keyFrame
    if(c1 && (c2 || c3))
    {
        //! create new keyframe
        KeyFrame::Ptr new_keyframe = KeyFrame::create(current_frame_);
        for(const Feature::Ptr &ft : fts)
        {
            if(ft->mpt_->isBad())
            {
                current_frame_->removeFeature(ft);
                continue;
            }

            ft->mpt_->addObservation(new_keyframe, ft);
            ft->mpt_->updateViewAndDepth();
            mapper_->addOptimalizeMapPoint(ft->mpt_);
        }
        new_keyframe->updateConnections();
        reference_keyframe_ = new_keyframe;
        last_keyframe_ = new_keyframe;
//        LOG(ERROR) << "C: (" << c1 << ", " << c2 << ", " << c3 << ") cur_n: " << current_frame_->N() << " ck: " << reference_keyframe_->N();
        return true;
    }
        //! change reference keyframe
    else
    {
        if(overlap_kfs[reference_keyframe_] < max_overlap * 0.85)
            reference_keyframe_ = max_overlap_keyframe;
        return false;
    }
}

void System::finishFrame()
{
    Stage last_stage = stage_;
    if(STAGE_NORMAL_FRAME == stage_)
    {
        if(STATUS_TRACKING_GOOD == status_)
        {

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

    LOG(WARNING) << "[System] Finish Current Frame with Stage: " << stage_;

    //! display
    viewer_->setCurrentFrame(current_frame_);
    showImage(last_stage);
}

void System::showImage(Stage stage)
{
//    cv::Mat image = rgb_;
//    if(image.channels() < 3)
//        cv::cvtColor(image, image, CV_GRAY2RGB);

    cv::Mat image;
    if(STAGE_NORMAL_FRAME == stage)
    {
        drowTrackedPoints(current_frame_, image);
    }
    else if(STAGE_INITALIZE == stage)
    {
        initializer_->drowOpticalFlow(image);
    }

    cv::imshow("SSVO Current Image", image);
    cv::waitKey(time_);
}

void System::drowTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst)
{
    //! draw features
    const cv::Mat src = frame->getImage(0);
    std::vector<Feature::Ptr> fts;
    frame->getFeatures(fts);
    cv::cvtColor(src, dst, CV_GRAY2RGB);
    int font_face = 1;
    double font_scale = 0.5;
    for(const Feature::Ptr &ft : fts)
    {
        Vector2d ft_px = ft->px_;
        cv::Point2f px(ft_px[0], ft_px[1]);
        cv::Scalar color(0, 255, 0);
        cv::circle(dst, px, 2, color, -1);

        string id_str = std::to_string((frame->Tcw()*ft->mpt_->pose()).norm());//ft->mpt_->getFoundRatio());//
        cv::putText(dst, id_str, px-cv::Point2f(1,1), font_face, font_scale, color);
    }

    //! draw seeds
    std::vector<Feature::Ptr> seed_fts;
    frame->getSeeds(seed_fts);
    for(const Feature::Ptr &ft : seed_fts)
    {
        Seed::Ptr seed = ft->seed_;
        cv::Point2f px(ft->px_[0], ft->px_[1]);
        double convergence = seed->z_range/std::sqrt(seed->sigma2);
        double scale = MIN(convergence, 256.0) / 256.0;
        cv::Scalar color(255*scale, 0, 255*(1-scale));
        cv::circle(dst, px, 2, color, -1);

//        string id_str = std::to_string();
//        cv::putText(dst, id_str, px-cv::Point2f(1,1), font_face, font_scale, color);
    }

}

void System::saveTrajectoryTUM(const std::string &file_name)
{
    std::ofstream f;
    f.open(file_name.c_str());
    f << std::fixed;

    std::list<double>::iterator frame_timestamp_ptr = frame_timestamp_buffer_.begin();
    std::list<Sophus::SE3d>::iterator frame_pose_ptr = frame_pose_buffer_.begin();
    std::list<KeyFrame::Ptr>::iterator reference_keyframe_ptr = reference_keyframe_buffer_.begin();
    const std::list<double>::iterator frame_timestamp = frame_timestamp_buffer_.end();
    for(; frame_timestamp_ptr!= frame_timestamp; frame_timestamp_ptr++, frame_pose_ptr++, reference_keyframe_ptr++)
    {
        Sophus::SE3d frame_pose = (*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
        Vector3d t = frame_pose.translation();
        Quaterniond q = frame_pose.unit_quaternion();

        f << std::setprecision(6) << *frame_timestamp_ptr << " "
          << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    f.close();
    LOG(INFO) << " Trajectory saved!";
}

}

