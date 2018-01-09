#include "feature_tracker.hpp"
#include "feature_alignment.hpp"
#include "image_alignment.hpp"

namespace ssvo{

FeatureTracker::FeatureTracker(int width, int height, int grid_size, int border, bool report, bool verbose) :
    report_(report), verbose_(report&&verbose)
{
    options_.border = border;
    options_.max_kfs = 10;
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
    std::set<KeyFrame::Ptr> connected_keyframes = frame->getRefKeyFrame()->getConnectedKeyFrames(options_.max_kfs);
    connected_keyframes.insert(frame->getRefKeyFrame());
    candidate_keyframes = connected_keyframes;

    for(const KeyFrame::Ptr &kf : connected_keyframes)
    {
        std::set<KeyFrame::Ptr> sub_connected_keyframe = kf->getConnectedKeyFrames(options_.max_kfs);
        for(const KeyFrame::Ptr &sub_kf : sub_connected_keyframe)
        {
            candidate_keyframes.insert(sub_kf);
        }
    }

    LOG_IF(INFO, verbose_) << "[FtTrack][2] -- Reproject Map Points --";
    double t1 = (double)cv::getTickCount();
    for(const KeyFrame::Ptr &kf : candidate_keyframes)
    {
        const std::vector<Feature::Ptr> &fts = kf->getFeatures();
        for(const Feature::Ptr &ft : fts)
        {
            if(ft->mpt == nullptr)
                continue;

            if(frame->isVisiable(ft->mpt->pose()))
                reprojectMapPoint(frame, ft->mpt);
        }
    }

    LOG_IF(INFO, verbose_) << "[FtTrack][3] -- Tracking Map Points --";
    double t2 = (double)cv::getTickCount();
    int matches = 0;
    total_project_ = 0;
    for(int index : grid_.grid_order)
    {
        if(trackMapPoints(frame, *grid_.cells[index]))
            matches++;
        if(matches > 120)
            break;
        }

    double t3 = (double)cv::getTickCount();
    LOG_IF(WARNING, report_) << "[FtTrack][*] Time: "
                           << (t1-t0)/cv::getTickFrequency() << " "
                           << (t2-t1)/cv::getTickFrequency() << " "
                           << (t3-t2)/cv::getTickFrequency() << " "
                           << ", match points " << matches << "(" << total_project_ << ")";

    // TODO 最后可以做一个对极线外点检测？

    return matches;
}

void FeatureTracker::resetGrid()
{
    for(Grid::Cell* c : grid_.cells) { c->clear(); }
    std::random_shuffle(grid_.grid_order.begin(), grid_.grid_order.end());
}

bool FeatureTracker::reprojectMapPoint(const Frame::Ptr &frame, const MapPoint::Ptr &point)
{
    Vector2d px(frame->cam_->project(frame->Tcw()*point->pose()));
    if(!frame->cam_->isInFrame(px.cast<int>(), options_.border))
        return false;

    const int k = static_cast<int>(px[1]/grid_.grid_size)*grid_.grid_n_cols
                + static_cast<int>(px[0]/grid_.grid_size);
    grid_.cells.at(k)->push_back(Candidate(point, px));

    return true;
}

bool FeatureTracker::trackMapPoints(const Frame::Ptr &frame, Grid::Cell &cell)
{
    // TODO sort? 选择质量较好的点优先投影
    cell.sort([](Candidate &c1, Candidate &c2){return c1.pt->getFoundRatio() > c2.pt->getFoundRatio();});

    for(Candidate candidate : cell)
    {
        double t0 = (double)cv::getTickCount();
        KeyFrame::Ptr kf_ref;
        int track_level = 0;
        if(!candidate.pt->getCloseViewObs(frame, kf_ref, track_level))
            continue;

        double t1 = (double)cv::getTickCount();
        const MapPoint::Ptr mpt = candidate.pt;
        const Feature::Ptr ft_ref = mpt->findObservation(kf_ref);
        const Vector3d obs_ref_dir(kf_ref->pose().translation() - mpt->pose());
        const SE3d T_cur_from_ref = frame->Tcw() * kf_ref->pose();
        const int patch_size = AlignPatch::Size;
        Matrix2d A_cur_from_ref;
        utils::getWarpMatrixAffine(kf_ref->cam_, frame->cam_, ft_ref->px, ft_ref->fn, ft_ref->level,
                                   obs_ref_dir.norm(), T_cur_from_ref, patch_size, A_cur_from_ref);

        // TODO 如果Affine很小的话，则不用warp
        const int patch_border_size = AlignPatch::SizeWithBorder;
        Matrix<float, patch_border_size, patch_border_size, RowMajor> patch_with_border;
        utils::warpAffine<float, patch_border_size>(kf_ref->getImage(ft_ref->level), patch_with_border, A_cur_from_ref,
                                                     ft_ref->px, ft_ref->level, track_level);

        double t2 = (double)cv::getTickCount();
        const cv::Mat image_cur = frame->getImage(track_level);

        const double factor = static_cast<double>(1 << track_level);
        Vector3d estimate(0,0,0); estimate.head<2>() = candidate.px / factor;

        static bool show = false;
        if(show)
        {
            cv::Mat show_ref, show_cur;
            cv::cvtColor(kf_ref->getImage(ft_ref->level), show_ref, CV_GRAY2RGB);
            cv::cvtColor(frame->getImage(track_level), show_cur, CV_GRAY2RGB);
            cv::circle(show_ref, cv::Point2i((int)ft_ref->px[0], (int)ft_ref->px[1])/(1 << ft_ref->level), 3, cv::Scalar(255,0,0));
            cv::circle(show_cur, cv::Point2i((int)estimate[0], (int)estimate[1]), 3, cv::Scalar(255,0,0));
            cv::imshow("ref track", show_ref);
            cv::imshow("cur track", show_cur);
            cv::waitKey(0);
        }

        total_project_++;
        if(AlignPatch::align2DI(image_cur, patch_with_border, estimate, 30, 0.01, verbose_))
        {
            Vector2d new_px = estimate.head<2>() * factor;
            Vector3d new_ft = frame->cam_->lift(new_px);
            Feature::Ptr new_feature = Feature::create(new_px, new_ft, track_level, mpt);
            frame->addFeature(new_feature);
            mpt->increaseVisible();
            mpt->increaseFound();

//            LOG(INFO) << "Creat new feature in level: " << track_level;
//
//            cv::circle(show_cur, cv::Point2i((int)estimate[0], (int)estimate[1]), 3, cv::Scalar(0,255,0));
//            cv::imshow("ref track", show_ref);
//            cv::imshow("cur track", show_cur);
//            cv::waitKey(0);
            double t3 = (double)cv::getTickCount();
            LOG_IF(INFO, verbose_) << "-level:" << track_level
                                  << " Time(ms),find obs: " << (t1-t0)/cv::getTickFrequency() * 1000
                                  << " perwarp: " << (t2-t1)/cv::getTickFrequency() * 1000
                                  << " align: " << (t3-t2)/cv::getTickFrequency() * 1000;
            return true;
        }
        else
        {
            //! if this point is not near the border, increase visiable
            if(frame->cam_->isInFrame(candidate.px.cast<int>()/factor, (patch_size >> 1) + 2, track_level))
                mpt->increaseVisible();
        }
    }

    return false;
}

}