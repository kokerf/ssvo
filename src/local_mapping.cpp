#include "local_mapping.hpp"

namespace ssvo{

namespace utils {

template<typename T>
double normal_distribution(T x, T mu, T sigma)
{
    static const double inv_sqrt_2pi = 0.3989422804014327f;
    double a = (x - mu) / sigma;

    return inv_sqrt_2pi / sigma * std::exp(-0.5 * a * a);
}

}

//! Seed
int Seed::seed_counter = 0;

Seed::Seed(KeyFrame::Ptr kf, Feature::Ptr ftr, double depth_mean, double depth_min) :
    id(seed_counter++),
    kf(kf),
    ftr(ftr),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36)
{}

double Seed::computeTau(
    const SE3d& T_ref_cur,
    const Vector3d& f,
    const double z,
    const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(-a.dot(t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    return (z_plus - z); // tau
}

void Seed::update(const double x, const double tau2)
{
    double norm_scale = sqrt(sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;

    double s2 = 1./(1./sigma2 + 1./tau2);
    double m = s2*(mu/sigma2 + x/tau2);
    double C1 = a/(a+b) * utils::normal_distribution<double>(x, mu, norm_scale);
    double C2 = b/(a+b) * 1./z_range;
    double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    double f = C1*(a+1.)/(a+b+1.) + C2*a/(a+b+1.);
    double e = C1*(a+1.)*(a+2.)/((a+b+1.)*(a+b+2.))
        + C2*a*(a+1.0f)/((a+b+1.0f)*(a+b+2.0f));

    // update parameters
    double mu_new = C1*m+C2*mu;
    sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new;
    mu = mu_new;
    a = (e-f)/(f-e/f);
    b = a*(1.0f-f)/f;
}

//! LocalMapper
LocalMapper::LocalMapper(const FastDetector::Ptr &fast_detector, double fps, bool report) :
    fast_detector_(fast_detector), delay_(static_cast<int>(1000.0/fps)), report_(report)
{
    map_ = Map::create();
    mapping_thread_ = std::make_shared<std::thread>(std::bind(&LocalMapper::run, this));
}

void LocalMapper::insertNewFrame(Frame::Ptr frame, KeyFrame::Ptr keyframe, double mean_depth, double min_depth)
{
    map_->insertKeyFrame(keyframe);
    keyframe->updateConnections();

    LOG_ASSERT(frame != nullptr) << "Error input! Frame should not be null!";
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);

        frames_buffer_.emplace_back(frame, keyframe);
        depth_buffer_.emplace_back(mean_depth, min_depth);
        cond_process_.notify_one();
    }
}

void LocalMapper::createInitalMap(const Frame::Ptr &frame_ref, const Frame::Ptr &frame_cur, const std::vector<Vector3d> &points)
{
    map_->clear();

    //! create Key Frame
    KeyFrame::Ptr keyframe_ref = ssvo::KeyFrame::create(frame_ref);
    KeyFrame::Ptr keyframe_cur = ssvo::KeyFrame::create(frame_cur);

    std::vector<Feature::Ptr> fts_ref = keyframe_ref->getFeatures();
    std::vector<Feature::Ptr> fts_cur = keyframe_cur->getFeatures();

    const size_t N = fts_ref.size();
    LOG_ASSERT(N == fts_cur.size()) << "Error in create inital map! Two frames' features is not matched!";
    LOG_ASSERT(N == points.size()) << "Error in create inital map! Two frames' features is not matched mappoints!";
    for(size_t i = 0; i < N; i++)
    {
        MapPoint::Ptr mpt = ssvo::MapPoint::create(points[i], keyframe_ref);

        fts_ref[i]->mpt = mpt;
        fts_cur[i]->mpt = mpt;

        map_->insertMapPoint(mpt);

        mpt->addObservation(keyframe_ref, fts_ref[i]);
        mpt->addObservation(keyframe_cur, fts_cur[i]);
        mpt->updateViewAndDepth();
    }

    Vector2d mean_depth, min_depth;
    keyframe_ref->getSceneDepth(mean_depth[0], min_depth[0]);
    keyframe_cur->getSceneDepth(mean_depth[1], min_depth[1]);
    this->insertNewFrame(frame_ref, keyframe_ref, mean_depth[0], min_depth[0]);
    this->insertNewFrame(frame_cur, keyframe_cur, mean_depth[1], min_depth[1]);

    LOG_IF(INFO, report_) << "[Mapping] Creating inital map with " << map_->MapPointsInMap() << " map points";
}

void LocalMapper::run()
{
    while(true)
    {
        if(!checkNewFrame())
            continue;

        processNewFrame();

        processNewKeyFrame();
    }
}

bool LocalMapper::checkNewFrame()
{
    {
        std::unique_lock<std::mutex> lock(mutex_kfs_);
        cond_process_.wait_for(lock, std::chrono::milliseconds(delay_));
        if(frames_buffer_.empty())
            return false;

        current_frame_ = frames_buffer_.front();
        current_depth_ = depth_buffer_.front();
        frames_buffer_.pop_front();
        depth_buffer_.pop_front();
    }

    return true;
}

bool LocalMapper::processNewKeyFrame()
{
    if(current_frame_.second == nullptr)
        return false;

    const KeyFrame::Ptr &kf = current_frame_.second;
    const double mean_depth = current_depth_.first;
    const double min_depth = current_depth_.second;
    std::vector<Feature::Ptr> fts = kf->getFeatures();

    Corners old_corners;
    old_corners.reserve(fts.size());
    for(const Feature::Ptr &ft : fts)
    {
        old_corners.emplace_back(Corner(ft->px[0], ft->px[1], 0, ft->level));
    }

    Corners new_corners;
    fast_detector_->detect(kf->image(), new_corners, old_corners, 150);

    for(const Corner &corner : new_corners)
    {
        const Vector2d px(corner.x, corner.y);
        const Vector3d fn(kf->cam_->lift(px));
        Feature::Ptr ft = Feature::create(px, fn, corner.level, nullptr);
        seeds_.emplace_back(Seed(kf, ft, mean_depth, min_depth));
    }

    LOG(INFO) << "[Mapping] Add new keyframe " << kf->id_ << " with " << new_corners.size() << " seeds";

    return true;
}

bool LocalMapper::processNewFrame()
{
    if(current_frame_.first == nullptr)
        return false;

    return true;
}


}