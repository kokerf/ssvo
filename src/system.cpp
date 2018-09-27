#include "config.hpp"
#include "system.hpp"
#include "optimizer.hpp"
#include "image_alignment.hpp"
#include "feature_alignment.hpp"
#include "time_tracing.hpp"

namespace ssvo{

std::string Config::file_name_;

TimeTracing::Ptr sysTrace = nullptr;

System::System(std::string config_file, std::string calib_flie) :
    stage_(STAGE_INITALIZE), status_(STATUS_INITAL_RESET),
    last_frame_(nullptr), current_frame_(nullptr), reference_keyframe_(nullptr)
{
    LOG_ASSERT(!calib_flie.empty()) << "Empty Calibration file input!!!";
    LOG_ASSERT(!config_file.empty()) << "Empty Config file input!!!";
    Config::file_name_ = config_file;

    AbstractCamera::Model model = AbstractCamera::checkCameraModel(calib_flie);
    if(AbstractCamera::Model::PINHOLE == model)
    {
        PinholeCamera::Ptr pinhole_camera = PinholeCamera::create(calib_flie);
        camera_ = std::static_pointer_cast<AbstractCamera>(pinhole_camera);
    }
    else if(AbstractCamera::Model::ATAN == model)
    {
        AtanCamera::Ptr atan_camera = AtanCamera::create(calib_flie);
        camera_ = std::static_pointer_cast<AbstractCamera>(atan_camera);
    }
    else
    {
        LOG(FATAL) << "Error camera model: " << model;
    }

    double fps = camera_->fps();
    if(fps < 1.0) fps = 1.0;
    //! image
    const int nlevels = Config::imageNLevels();
    const double scale_factor = Config::imageScaleFactor();
    const int width = camera_->width();
    const int height = camera_->height();
    const int image_border = AlignPatch::Size;
    //! corner detector
    const int grid_size = Config::gridSize();
    const int grid_min_size = Config::gridMinSize();
    const int fast_max_threshold = Config::fastMaxThreshold();
    const int fast_min_threshold = Config::fastMinThreshold();

    fast_detector_ = FastDetector::create(width, height, image_border, nlevels, scale_factor, grid_size, grid_min_size, fast_max_threshold, fast_min_threshold);
    feature_tracker_ = FeatureTracker::create(width, height, 20, image_border, true);
    initializer_ = Initializer::create(fast_detector_, true);
    mapper_ = LocalMapper::create(fast_detector_, true, false);
    depth_filter_ = DepthFilter::create(fast_detector_, true);
    DepthFilter::SeedCallback seed_converge_callback = std::bind(&LocalMapper::createFeatureFromSeed, mapper_, std::placeholders::_1);
    DepthFilter::KeyFrameCallback keyframe_callback = std::bind(&LocalMapper::insertKeyFrame, mapper_, std::placeholders::_1);
    depth_filter_->setSeedConvergedCallback(seed_converge_callback);
    depth_filter_->setKeyFrameProcessCallback(keyframe_callback);

    viewer_ = Viewer::create(mapper_->map_, cv::Size(width, height));
    //viewer_->setShowFalg(false);

    Frame::initScaleParameters(fast_detector_);

    //mapper_->startMainThread();
    //depth_filter_->startMainThread();

    time_ = 1000.0/fps;

    options_.min_kf_disparity = 100;//MIN(Config::imageHeight(), Config::imageWidth())/5;
    options_.min_ref_track_rate = 0.7;

    //! LOG and timer for system;
    TimeTracing::TraceNames time_names;
    time_names.push_back("total");
    time_names.push_back("processing");
    time_names.push_back("frame_create");
    time_names.push_back("img_align");
    time_names.push_back("feature_reproj");
    time_names.push_back("motion_ba");
    time_names.push_back("light_affine");
    time_names.push_back("per_depth_filter");
    time_names.push_back("finish");

    TimeTracing::TraceNames log_names;
    log_names.push_back("frame_id");
    log_names.push_back("num_feature_reproj");
    log_names.push_back("stage");

    string trace_dir = Config::timeTracingDirectory();
    sysTrace.reset(new TimeTracing("ssvo_trace_system", trace_dir, time_names, log_names));
}

System::~System()
{
    sysTrace.reset();

    viewer_->setStop();
    //depth_filter_->stopMainThread();
    mapper_->stopMainThread();

    viewer_->waitForFinish();
}

void System::process(const cv::Mat &image, const double timestamp)
{
    sysTrace->startTimer("total");
    sysTrace->startTimer("frame_create");
    //! get gray image
    double t0 = (double)cv::getTickCount();
    rgb_ = image;
    cv::Mat gray = image.clone();
    if(gray.channels() == 3)
        cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);

    current_frame_ = Frame::create(gray, timestamp, camera_);
    double t1 = (double)cv::getTickCount();
    LOG(WARNING) << "[System] Frame " << current_frame_->id_ << " create time: " << (t1-t0)/cv::getTickFrequency();
    sysTrace->log("frame_id", current_frame_->id_);
    sysTrace->stopTimer("frame_create");

    sysTrace->startTimer("processing");
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
        status_ = relocalize();
    }
    sysTrace->stopTimer("processing");

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
    createInitalMap(initializer_->getReferenceFrame(), current_frame_);

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

    initializer_->reset();

    return STATUS_INITAL_SUCCEED;
}

System::Status System::tracking()
{
    current_frame_->setRefKeyFrame(reference_keyframe_);

    //! track seeds
    //depth_filter_->trackFrame(last_frame_, current_frame_);

    // TODO 先验信息怎么设置？
    current_frame_->setPose(last_frame_->pose());
    //! alignment by SE3
    AlignSE3 align;
    sysTrace->startTimer("img_align");
    int obs = align.run(last_frame_, current_frame_, Config::imageNLevels()-1, Config::alignBottomLevel(), Config::alignScaleFactor(), 30, 1e-8);
    sysTrace->stopTimer("img_align");

    //! track local map
    sysTrace->startTimer("feature_reproj");
    int matches = feature_tracker_->reprojectLoaclMap(current_frame_);
    sysTrace->stopTimer("feature_reproj");
    sysTrace->log("num_feature_reproj", matches);
    LOG(WARNING) << "[System] Track with " << obs << ", "<< matches << " points";

    // TODO tracking status
    if(matches < Config::minQualityFts())
        return STATUS_TRACKING_BAD;

    //! motion-only BA
    sysTrace->startTimer("motion_ba");
    Optimizer::motionOnlyBundleAdjustment(current_frame_, false, false, true);
    sysTrace->stopTimer("motion_ba");

    sysTrace->startTimer("per_depth_filter");
    bool keyframe_created = createNewKeyFrame();
    depth_filter_->insertFrame(current_frame_, keyframe_created ? reference_keyframe_: nullptr);
    sysTrace->stopTimer("per_depth_filter");

    sysTrace->startTimer("light_affine");
    calcLightAffine();
    sysTrace->stopTimer("light_affine");

    //！ save frame pose
    frame_timestamp_buffer_.push_back(current_frame_->timestamp_);
    reference_keyframe_buffer_.push_back(current_frame_->getRefKeyFrame());
    frame_pose_buffer_.push_back(current_frame_->pose());//current_frame_->getRefKeyFrame()->Tcw() * current_frame_->pose());

    return STATUS_TRACKING_GOOD;
}

System::Status System::relocalize()
{
    Corners corners_new;
    Corners corners_old;
    fast_detector_->detect(current_frame_->images(), corners_new, corners_old, Config::minCornersPerKeyFrame());

    reference_keyframe_ = mapper_->relocalizeByDBoW(current_frame_, corners_new);

    if(reference_keyframe_ == nullptr)
        return STATUS_TRACKING_BAD;

    current_frame_->setPose(reference_keyframe_->pose());

    //! alignment by SE3
    AlignSE3 align;
    int matches = align.run(reference_keyframe_, current_frame_, Config::imageNLevels()-1, Config::alignBottomLevel(), Config::alignScaleFactor(), 30, 1e-8);

    if(matches < 30)
        return STATUS_TRACKING_BAD;

    current_frame_->setRefKeyFrame(reference_keyframe_);
    matches = feature_tracker_->reprojectLoaclMap(current_frame_);

    if(matches < 30)
        return STATUS_TRACKING_BAD;

    Optimizer::motionOnlyBundleAdjustment(current_frame_, false, true, true);

    if(current_frame_->getMapPointMatchSize() < 30)
        return STATUS_TRACKING_BAD;

    return STATUS_TRACKING_GOOD;
}

void System::calcLightAffine()
{
    const std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts_last = last_frame_->getMapPointFeaturesMatched();
    const std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts_curr = current_frame_->getMapPointFeaturesMatched();

    const cv::Mat img_last = last_frame_->getImage(0);
    const cv::Mat img_curr = current_frame_->getImage(0);

    const int size = 4;
    const int patch_area = size*size;
    const int N = (int)mpt_fts_last.size();
    cv::Mat patch_buffer_last = cv::Mat::zeros(N, patch_area, CV_32FC1);
    cv::Mat patch_buffer_curr = cv::Mat::zeros(N, patch_area, CV_32FC1);

    int count = 0;
    for(const auto &mpt_ft : mpt_fts_last)
    {
        const MapPoint::Ptr mpt = mpt_ft.first;
        const Feature::Ptr ft_last = mpt_ft.second;

        const auto itr = mpt_fts_curr.find(mpt);
        if(itr == mpt_fts_curr.end())
            continue;

        const Feature::Ptr ft_curr = itr->second;

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

void System::createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur)
{
    mapper_->clear();

    //! create Key Frame
    KeyFrame::Ptr keyframe_ref = KeyFrame::create(frame_ref);
    KeyFrame::Ptr keyframe_cur = KeyFrame::create(frame_cur);

    //! create seeds
    depth_filter_->insertKeyFrame(keyframe_ref);
    depth_filter_->insertKeyFrame(keyframe_cur);

    //! add to map
    mapper_->createInitalMap(keyframe_ref, keyframe_cur);
}

bool System::createNewKeyFrame()
{
    std::map<KeyFrame::Ptr, int> overlap_kfs = current_frame_->getOverLapKeyFrames();

    std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts = current_frame_->getMapPointFeaturesMatched();

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
        ;//c1 = false;

    //! check disparity
    std::list<float> disparities;
    const int threahold = int (max_overlap * 0.6);
    for(const auto &ovlp_kf : overlap_kfs)
    {
        if(ovlp_kf.second < threahold)
            continue;

        std::vector<float> disparity;
        disparity.reserve(ovlp_kf.second);
        std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts_kf = ovlp_kf.first->getMapPointFeaturesMatched();
        for(const auto &item : mpt_fts_kf)
        {
            const MapPoint::Ptr &mpt = item.first;

            if(!mpt_fts.count(mpt)) continue;

            Feature::Ptr ft_cur = mpt_fts.find(mpt)->second;
            Feature::Ptr ft_ref = item.second;

            const Vector2d px(ft_ref->px_ - ft_cur->px_);
            disparity.push_back(px.norm());
        }

        std::sort(disparity.begin(), disparity.end());
        float disp = disparity.at(disparity.size()/2);
        disparities.push_back(disp);
    }
    disparities.sort();

    LOG(INFO) << "[System] Max overlap: " << max_overlap << " min disaprity " << disparities.front();

//    int all_features = current_frame_->featureNumber() + current_frame_->seedNumber();
    bool c2 = disparities.front() > options_.min_kf_disparity;
    bool c3 = current_frame_->getMapPointMatchSize() < reference_keyframe_->getMapPointMatchSize() * options_.min_ref_track_rate;
//    bool c4 = current_frame_->featureNumber() < reference_keyframe_->featureNumber() * 0.9;

    //! create new keyFrame
    if(c1 && (c2 || c3))
    {
        //! create new keyframe
        KeyFrame::Ptr new_keyframe = KeyFrame::create(current_frame_);
        const std::vector<MapPoint::Ptr> mpts = new_keyframe->getMapPoints();
        const std::vector<size_t> matches = new_keyframe->getMapPointMatchIndices();
        for(const size_t &idx : matches)
        {
            const MapPoint::Ptr &mpt = mpts[idx];

            mpt->addObservation(new_keyframe, idx);
            mpt->updateViewAndDepth();
//            mapper_->addOptimalizeMapPoint(ft->mpt_);
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
    sysTrace->startTimer("finish");
    cv::Mat image_show;
//    Stage last_stage = stage_;
    if(STAGE_NORMAL_FRAME == stage_)
    {
        if(STATUS_TRACKING_BAD == status_)
        {
            stage_ = STAGE_RELOCALIZING;
            current_frame_->setPose(last_frame_->pose());
        }
    }
    else if(STAGE_INITALIZE == stage_)
    {
        if(STATUS_INITAL_SUCCEED == status_)
            stage_ = STAGE_NORMAL_FRAME;
        else if(STATUS_INITAL_RESET == status_)
            initializer_->reset();

        initializer_->drowOpticalFlow(image_show);
    }
    else if(STAGE_RELOCALIZING == stage_)
    {
        if(STATUS_TRACKING_GOOD == status_)
            stage_ = STAGE_NORMAL_FRAME;
        else
            current_frame_->setPose(last_frame_->pose());
    }

    //! update
    last_frame_ = current_frame_;

    //! display
    viewer_->setCurrentFrame(current_frame_, image_show);

    sysTrace->log("stage", stage_);
    sysTrace->stopTimer("finish");
    sysTrace->stopTimer("total");
    const double time = sysTrace->getTimer("total");
    LOG(WARNING) << "[System] Finish Current Frame with Stage: " << stage_ << ", total time: " << time;

    sysTrace->writeToFile();

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
        Sophus::SE3d frame_pose = (*frame_pose_ptr);//(*reference_keyframe_ptr)->Twc() * (*frame_pose_ptr);
        Vector3d t = frame_pose.translation();
        Quaterniond q = frame_pose.unit_quaternion();

        f << std::setprecision(6) << *frame_timestamp_ptr << " "
          << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    f.close();
    LOG(INFO) << " Trajectory saved!";
}

}

