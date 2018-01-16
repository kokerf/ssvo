#include "feature_tracker.hpp"
#include "feature_alignment.hpp"
#include "image_alignment.hpp"

namespace ssvo{

FeatureTracker::FeatureTracker(int width, int height, int grid_size, int border, bool report, bool verbose) :
    report_(report), verbose_(report&&verbose)
{
    options_.border = border;
    options_.max_matches = 200;
    //! initialize grid
    grid_.grid_size = grid_size;
    grid_.grid_n_cols = ceil(static_cast<double>(width)/grid_.grid_size);
    grid_.grid_n_rows = ceil(static_cast<double>(height)/grid_.grid_size);
    grid_.cells.resize(grid_.grid_n_rows*grid_.grid_n_cols);
    for(Grid::Cell*& c : grid_.cells) { c = new Grid::Cell; }
    grid_.grid_order.resize(grid_.cells.size());
    for(size_t i = 0; i < grid_.grid_order.size(); ++i) { grid_.grid_order[i] = i; }
    std::random_shuffle(grid_.grid_order.begin(), grid_.grid_order.end());
}

FeatureTracker::~FeatureTracker()
{
    for(Grid::Cell*& c : grid_.cells) { delete c; }
}

int FeatureTracker::reprojectLoaclMap(const Frame::Ptr &frame)
{
    LOG_IF(INFO, verbose_) << "[FtTrack][*] -- Reproject Local Map --";
    double t0 = (double)cv::getTickCount();

    resetGrid();

    LOG_IF(INFO, verbose_) << "[FtTrack][1] -- Get Candidate KeyFrame --";
    std::set<KeyFrame::Ptr> candidate_keyframes;
    std::set<KeyFrame::Ptr> connected_keyframes = frame->getRefKeyFrame()->getConnectedKeyFrames();
    connected_keyframes.insert(frame->getRefKeyFrame());
    candidate_keyframes = connected_keyframes;

    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        std::set<KeyFrame::Ptr> sub_connected_keyframe = kf->getConnectedKeyFrames();
        for(const KeyFrame::Ptr &sub_kf : sub_connected_keyframe)
        {
            candidate_keyframes.insert(sub_kf);
        }
    }

    LOG_IF(INFO, verbose_) << "[FtTrack][2] -- Reproject Map Points --";
    double t1 = (double)cv::getTickCount();
    std::unordered_set<MapPoint::Ptr> local_mpts;
    for(const KeyFrame::Ptr &kf : candidate_keyframes)
    {
        MapPoints mpts;
        kf->getMapPoints(mpts);
        for(const MapPoint::Ptr &mpt : mpts)
        {
            if(local_mpts.count(mpt))
                continue;

            local_mpts.insert(mpt);

            if(mpt->isBad()) //! should not happen
            {
                kf->removeMapPoint(mpt);
                continue;
            }

            reprojectMapPointToCell(frame, mpt);
        }
    }

    LOG_IF(INFO, verbose_) << "[FtTrack][3] -- Tracking Map Points --";
    double t2 = (double)cv::getTickCount();
    int matches = 0;
    total_project_ = 0;
    for(int index : grid_.grid_order)
    {
        if(matchMapPointsFromCell(frame, *grid_.cells[index]))
            matches++;
        if(matches > options_.max_matches)
            break;
    }

    double t3 = (double)cv::getTickCount();
    LOG_IF(WARNING, report_) << "[FtTrack][*] Time: "
                           << (t1-t0)/cv::getTickFrequency() << " "
                           << (t2-t1)/cv::getTickFrequency() << " "
                           << (t3-t2)/cv::getTickFrequency() << " "
                           << ", match points " << matches << "(" << total_project_ << ", " << local_mpts.size() << ")";

    // TODO 最后可以做一个对极线外点检测？

    return matches;
}

void FeatureTracker::resetGrid()
{
    for(Grid::Cell* c : grid_.cells) { c->clear(); }
    std::random_shuffle(grid_.grid_order.begin(), grid_.grid_order.end());
}

bool FeatureTracker::reprojectMapPointToCell(const Frame::Ptr &frame, const MapPoint::Ptr &point)
{
    Vector3d pose(frame->Tcw() * point->pose());
    if(pose[2] < 0.0f)
        return false;

    Vector2d px(frame->cam_->project(pose));
    if(!frame->cam_->isInFrame(px.cast<int>(), options_.border))
        return false;

    const int k = static_cast<int>(px[1]/grid_.grid_size)*grid_.grid_n_cols
                + static_cast<int>(px[0]/grid_.grid_size);
    grid_.cells.at(k)->push_back(Candidate(point, px));

    return true;
}

bool FeatureTracker::matchMapPointsFromCell(const Frame::Ptr &frame, Grid::Cell &cell)
{
    // TODO sort? 选择质量较好的点优先投影
    cell.sort([](Candidate &c1, Candidate &c2){return c1.pt->getFoundRatio() > c2.pt->getFoundRatio();});

    for(const Candidate &candidate : cell)
    {
        total_project_++;
        const MapPoint::Ptr &mpt = candidate.pt;
        Vector2d px_cur = candidate.px;
        int level_cur = 0;
        int result = reprojectMapPoint(frame, mpt, px_cur, level_cur, 30, 0.01, verbose_);

        mpt->increaseVisible(result+1);

        if(result != 1)
            continue;

        Vector3d ft_cur = frame->cam_->lift(px_cur);
        Feature::Ptr new_feature = Feature::create(px_cur, ft_cur, level_cur, mpt);
        frame->addFeature(new_feature);
        mpt->increaseFound(2);

        return true;
    }

    return false;
}

int FeatureTracker::reprojectMapPoint(const Frame::Ptr &frame,
                                       const MapPoint::Ptr &mpt,
                                       Vector2d &px_cur,
                                       int &level_cur,
                                       const int max_iterations,
                                       const double epslion,
                                       bool verbose)
{
    static const int patch_size = AlignPatch::Size;
    static const int patch_border_size = AlignPatch::SizeWithBorder;

    KeyFrame::Ptr kf_ref;
    if(!mpt->getCloseViewObs(frame, kf_ref, level_cur))
        return -1;

    const Feature::Ptr ft_ref = mpt->findObservation(kf_ref);
    const Vector3d obs_ref_dir(kf_ref->pose().translation() - mpt->pose());
    const SE3d T_cur_from_ref = frame->Tcw() * kf_ref->pose();

    Matrix2d A_cur_from_ref;
    utils::getWarpMatrixAffine(kf_ref->cam_, frame->cam_, ft_ref->px_, ft_ref->fn_, ft_ref->level_,
                               obs_ref_dir.norm(), T_cur_from_ref, patch_size, A_cur_from_ref);

    // TODO 如果Affine很小的话，则不用warp
    const cv::Mat image_ref = kf_ref->getImage(ft_ref->level_);
    Matrix<float, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<float, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                ft_ref->px_, ft_ref->level_, level_cur);

    const cv::Mat image_cur = frame->getImage(level_cur);

    const double factor = static_cast<double>(1 << level_cur);
    Vector3d estimate(0,0,0); estimate.head<2>() = px_cur / factor;

    static bool show = false;
    if(show)
    {
        cv::Mat show_ref, show_cur;
        cv::cvtColor(image_ref, show_ref, CV_GRAY2RGB);
        cv::cvtColor(image_cur, show_cur, CV_GRAY2RGB);
        cv::circle(show_ref, cv::Point2i((int)ft_ref->px_[0], (int)ft_ref->px_[1])/(1 << ft_ref->level_), 3, cv::Scalar(255,0,0));
        cv::circle(show_cur, cv::Point2i((int)estimate[0], (int)estimate[1]), 3, cv::Scalar(255,0,0));
        cv::imshow("ref track", show_ref);
        cv::imshow("cur track", show_cur);
        cv::waitKey(0);
    }

    bool matched = AlignPatch::align2DI(image_cur, patch_with_border, estimate, max_iterations, epslion, verbose);
    if(!matched)
        return 0;

    px_cur = estimate.head<2>() * factor;

    return 1;
}

bool FeatureTracker::trackFeature(const Frame::Ptr &frame_ref,
                                  const Frame::Ptr &frame_cur,
                                  const Feature::Ptr &ft_ref,
                                  Vector2d &px_cur,
                                  int &level_cur,
                                  const int max_iterations,
                                  const double epslion,
                                  bool verbose)
{
    static const int patch_size = AlignPatch::Size;
    static const int patch_border_size = AlignPatch::SizeWithBorder;

    const Vector3d obs_ref_dir(frame_ref->pose().translation() - ft_ref->mpt_->pose());
    const SE3d T_cur_from_ref = frame_cur->Tcw() * frame_ref->pose();

    Matrix2d A_cur_from_ref;
    utils::getWarpMatrixAffine(frame_ref->cam_, frame_cur->cam_, ft_ref->px_, ft_ref->fn_, ft_ref->level_,
                               obs_ref_dir.norm(), T_cur_from_ref, patch_size, A_cur_from_ref);

    level_cur = utils::getBestSearchLevel(A_cur_from_ref, frame_cur->max_level_);
//    std::cout << "A:\n" << A_cur_from_ref << std::endl;

    const cv::Mat image_ref = frame_ref->getImage(ft_ref->level_);
    Matrix<float, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<float, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                ft_ref->px_, ft_ref->level_, level_cur);

    const cv::Mat image_cur = frame_cur->getImage(level_cur);

    const double factor = static_cast<double>(1 << level_cur);
    Vector3d estimate(0,0,0); estimate.head<2>() = px_cur / factor;

    bool matched = AlignPatch::align2DI(image_cur, patch_with_border, estimate, max_iterations, epslion, verbose);
    if(!matched)
        return false;

    px_cur = estimate.head<2>() * factor;

    return true;
}

}