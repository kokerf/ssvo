#include "feature_tracker.hpp"
#include "alignment.hpp"
#include "utils.hpp"

namespace ssvo{

namespace utils{

void getWarpMatrixAffine(const Camera::Ptr &cam_ref,
                                const Camera::Ptr &cam_cur,
                                const Vector2d &px_ref,
                                const Vector3d &f_ref,
                                const int level_ref,
                                const double depth_ref,
                                const Sophus::SE3d &T_cur_ref,
                                const int patch_size,
                                Matrix2d &A_cur_ref)
{
    const double half_patch_size = static_cast<double>(patch_size+2)/2;
    const Vector3d xyz_ref(depth_ref * f_ref);
    const double length = half_patch_size * (1 << level_ref);
    Vector3d xyz_ref_du(cam_ref->lift(px_ref + Vector2d(length, 0)));
    Vector3d xyz_ref_dv(cam_ref->lift(px_ref + Vector2d(0, length)));
    xyz_ref_du *= xyz_ref[2]/xyz_ref_du[2];
    xyz_ref_dv *= xyz_ref[2]/xyz_ref_dv[2];
    const Vector2d px_cur(cam_cur->project(T_cur_ref * xyz_ref));
    const Vector2d px_du(cam_cur->project(T_cur_ref * xyz_ref_du));
    const Vector2d px_dv(cam_cur->project(T_cur_ref * xyz_ref_dv));
    A_cur_ref.col(0) = (px_du - px_cur)/half_patch_size;
    A_cur_ref.col(1) = (px_dv - px_cur)/half_patch_size;
}

template<typename Td, int size>
void warpAffine(const cv::Mat &img_ref,
                Matrix<Td, size, size, RowMajor> &patch,
               const Matrix2d &A_cur_from_ref,
               const Vector2d &px_ref,
               const int level_ref,
               const int level_cur)
{
    assert(img_ref.type() == CV_8UC1);

    const Matrix2d A_ref_from_cur = A_cur_from_ref.inverse();
    if(isnan(A_ref_from_cur(0,0)))
    {
        LOG(ERROR) << "Affine warp is Nan";
        return;
    }

    const Vector2d px_ref_pyr = px_ref / (1 << level_ref);
    const double half_patch_size = size * 0.5;
    const int px_pyr_scale = 1 << level_cur;
    for(int y = 0; y < size; ++y)
    {
        for (int x = 0; x < size; ++x)
        {
            Vector2d px_patch(x-half_patch_size, y-half_patch_size);
            px_patch *= px_pyr_scale;//! A_ref_from_cur is for level-0, so transform to it
            const Vector2d px(A_ref_from_cur*px_patch + px_ref_pyr);

            if(px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
                patch(y, x) = 0;
            else
                patch(y, x) = (Td) utils::interpolateMat_8u(img_ref, px[0], px[1]);
        }
    }
}

}

FeatureTracker::FeatureTracker(int width, int height, int grid_size, bool report, bool verbose) :
    report_(report), verbose_(verbose)
{
    options_.border = Align2DI::HalfPatchSize;
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

int FeatureTracker::reprojectLoaclMap(const Frame::Ptr &frame, const Map::Ptr &map)
{
    LOG_IF(INFO, report_) << "[FtTrack][*] -- Reproject Local Map --";
    double t0 = (double)cv::getTickCount();

    resetGrid();

    LOG_IF(INFO, report_) << "[FtTrack][1] -- Get Candidate KeyFrame --";
    std::vector<KeyFrame::Ptr> candidate_keyframes = frame->getRefKeyFrame()->getConnectedKeyFrames();
    if(candidate_keyframes.size() > options_.border-1) {
        candidate_keyframes.resize(options_.border-1);
    }

    candidate_keyframes.push_back(frame->getRefKeyFrame());

    LOG_IF(INFO, report_) << "[FtTrack][2] -- Reproject Map Points --";
    double t1 = (double)cv::getTickCount();
    for(KeyFrame::Ptr kf : candidate_keyframes)
    {
        for(Feature::Ptr ft : kf->features())
        {
            if(ft->mpt == nullptr) {
                continue;
            }

            if(frame->isVisiable(ft->mpt->pose()))
            {
                reprojectMapPoint(frame, ft->mpt);
            }
        }
    }

    LOG_IF(INFO, report_) << "[FtTrack][3] -- Tracking Map Points --";
    double t2 = (double)cv::getTickCount();
    int matches = 0;
    for(int index : grid_.grid_order)
    {
        if(trackMapPoints(frame, *grid_.cells[index])) {
            matches++;
        }

        if(matches > 120) {
            break;
        }
    }

    double t3 = (double)cv::getTickCount();
    LOG_IF(WARNING, report_) << "[FtTrack][*] Time: "
                           << (t1-t0)/cv::getTickFrequency() << " "
                           << (t2-t1)/cv::getTickFrequency() << " "
                           << (t3-t2)/cv::getTickFrequency() << " "
                           << ", match points " << matches;

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
        const Sophus::SE3d T_cur_from_ref = frame->Tcw() * kf_ref->pose();
        const int patch_size = Align2DI::PatchSize;
        Matrix2d A_cur_from_ref;
        utils::getWarpMatrixAffine(kf_ref->cam_, frame->cam_, ft_ref->px, ft_ref->fn, ft_ref->level,
                                   obs_ref_dir.norm(), T_cur_from_ref, patch_size, A_cur_from_ref);

        // TODO 如果Affine很小的话，则不用warp
        const int patch_border_size = patch_size+2;
        Matrix<double, patch_border_size, patch_border_size, RowMajor> patch_with_border;
        utils::warpAffine<double, patch_border_size>(kf_ref->getImage(ft_ref->level), patch_with_border, A_cur_from_ref,
                                                     ft_ref->px, ft_ref->level, track_level);

        Matrix<double, patch_size, patch_size, RowMajor> patch, dx, dy;
        patch = patch_with_border.block(1, 1, patch_size, patch_size);
        dx = 0.5*(patch_with_border.block(1, 2, patch_size, patch_size)
            - patch_with_border.block(1, 0, patch_size, patch_size));
        dy = 0.5*(patch_with_border.block(2, 1, patch_size, patch_size)
            - patch_with_border.block(0, 1, patch_size, patch_size));

        double t2 = (double)cv::getTickCount();
        const cv::Mat cur_image = frame->getImage(track_level);
        Eigen::Map<Matrix<uchar, Dynamic, Dynamic, RowMajor> > cur_eigen_image((uchar*)cur_image.data, cur_image.rows, cur_image.cols);
        Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > patch_eigen(patch.data(), patch_size*patch_size, 1);
        Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > dx_eigen(dx.data(), patch_size*patch_size, 1);
        Eigen::Map<Matrix<double, Dynamic, Dynamic, RowMajor> > dy_eigen(dy.data(), patch_size*patch_size, 1);

        Align2DI matcher(verbose_);
        const double factor = static_cast<double>(1 << track_level);
        Vector3d estimate(0,0,0); estimate.head<2>() = candidate.px / factor;

//        cv::Mat show_ref, show_cur;
//        cv::cvtColor(kf_ref->getImage(ft_ref->level), show_ref, CV_GRAY2RGB);
//        cv::cvtColor(cur_image, show_cur, CV_GRAY2RGB);
//        cv::circle(show_ref, cv::Point2i((int)ft_ref->px[0], (int)ft_ref->px[1])/(1 << ft_ref->level), 3, cv::Scalar(255,0,0));
//        cv::circle(show_cur, cv::Point2i((int)estimate[0], (int)estimate[1]), 3, cv::Scalar(255,0,0));
//        cv::imshow("ref track", show_ref);
//        cv::imshow("cur track", show_cur);
//        cv::waitKey(0);

        if(matcher.run(cur_eigen_image, patch_eigen, dx_eigen, dy_eigen, estimate))
        {
            Vector2d new_px = estimate.head<2>() * factor;
            Vector3d new_ft = frame->cam_->lift(new_px);
            Feature::Ptr new_feature = Feature::create(new_px, new_ft.normalized(), track_level, mpt);
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