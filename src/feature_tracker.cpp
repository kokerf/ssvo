#include "feature_tracker.hpp"
#include "feature_alignment.hpp"
#include "image_alignment.hpp"
#include "utils.hpp"
#include <opencv2/core/eigen.hpp>
#include <numeric>

namespace ssvo{

template <>
inline size_t Grid<std::pair<MapPoint::Ptr, Vector2d> >::getIndex(const std::pair<MapPoint::Ptr, Vector2d> &element)
{
    const Vector2d &px = element.second;
    return static_cast<size_t>(px[1]/grid_size_)*grid_n_cols_
        + static_cast<size_t>(px[0]/grid_size_);
}

FeatureTracker::FeatureTracker(int width, int height, int grid_size, int border, bool report, bool verbose) :
    grid_(width, height, grid_size), report_(report), verbose_(report&&verbose)
{
    options_.border = border;
    options_.max_matches = 200;
    options_.max_track_kfs = Config::maxTrackKeyFrames();
    options_.num_align_iter = 30;
    options_.max_align_epsilon = 0.01;
    options_.max_align_error2 = 3.0;

    //! initialize grid
    grid_order_.resize(grid_.nCells());
    std::iota(grid_order_.begin(), grid_order_.end(), 0);
}

int FeatureTracker::reprojectLoaclMap(const Frame::Ptr &frame)
{
    static Frame::Ptr frame_last;

    double t0 = (double)cv::getTickCount();

    grid_.clear();

    logs_.num_frame_matches = 0;
    logs_.num_cells_matches = 0;
    logs_.num_total_project = 0;
    logs_.num_local_mpts = 0;

    std::unordered_set<MapPoint::Ptr> last_mpts_set;
    if(frame_last)
    {
        logs_.num_frame_matches = matchMapPointsFromLastFrame(frame, frame_last);
        std::vector<MapPoint::Ptr> last_mpts_list = frame_last->getMapPoints();
        for(const MapPoint::Ptr &mpt : last_mpts_list)
        {
            if(mpt) last_mpts_set.insert(mpt);
        }
    }

    std::set<KeyFrame::Ptr> local_keyframes = frame->getRefKeyFrame()->getConnectedKeyFrames(options_.max_track_kfs);
    local_keyframes.insert(frame->getRefKeyFrame());

    if(local_keyframes.size() < options_.max_track_kfs)
    {
        std::set<KeyFrame::Ptr> sub_connected_keyframes = frame->getRefKeyFrame()->getSubConnectedKeyFrames(options_.max_track_kfs-local_keyframes.size());
        for(const KeyFrame::Ptr &kf : sub_connected_keyframes)
        {
            local_keyframes.insert(kf);
        }
    }

    double t1 = (double)cv::getTickCount();

    std::unordered_set<MapPoint::Ptr> local_mpts;
    for(const KeyFrame::Ptr &kf : local_keyframes)
    {
        const std::vector<size_t> matches = kf->getMapPointMatchIndices();
        const std::vector<MapPoint::Ptr> mpts = kf->getMapPoints();
        for(const size_t &idx : matches)
        {
            const MapPoint::Ptr mpt = mpts[idx];

            if(mpt->isBad()) //! should not happen
            {
                kf->removeMapPointMatchByIndex(idx);// TODO
                continue;
            }

            if(local_mpts.count(mpt) || last_mpts_set.count(mpt))
                continue;

            local_mpts.insert(mpt);

            reprojectMapPointToCell(frame, mpt);
        }
    }

    double t2 = (double)cv::getTickCount();
    const int max_matches_rest = options_.max_matches - logs_.num_frame_matches;

    std::random_shuffle(grid_order_.begin(), grid_order_.end());
    for(size_t index : grid_order_)
    {
        if(grid_.isMasked(index))
            continue;

        if(matchMapPointsFromCell(frame, grid_.getCell(index)))
            logs_.num_cells_matches++;

        if(logs_.num_cells_matches > max_matches_rest)
            break;
    }

    double t3 = (double)cv::getTickCount();

    logs_.time_frame_match = (t1 - t0) / cv::getTickFrequency();
    logs_.time_cells_create = (t2 - t1) / cv::getTickFrequency();
    logs_.time_cells_match = (t3 - t2) / cv::getTickFrequency();
    logs_.num_local_mpts = (int)local_mpts.size();

    //! update last frame
    frame_last = frame;

    return logs_.num_frame_matches + logs_.num_cells_matches;
}

bool FeatureTracker::reprojectMapPointToCell(const Frame::Ptr &frame, const MapPoint::Ptr &point)
{
    Vector3d pose(frame->Tcw() * point->pose());
    if(pose[2] < 0.0f)
        return false;

    Vector2d px(frame->cam_->project(pose));
    if(!frame->cam_->isInFrame(px.cast<int>(), options_.border))
        return false;

    grid_.insert(std::make_pair(point, px));

    return true;
}

bool FeatureTracker::matchMapPointsFromCell(const Frame::Ptr &frame, Grid<std::pair<MapPoint::Ptr, Vector2d>>::Cell &cell)
{
    // TODO sort? 选择质量较好的点优先投影
    cell.sort([](std::pair<MapPoint::Ptr, Vector2d> &ft1, std::pair<MapPoint::Ptr, Vector2d> &ft2){
      return ft1.first->getFoundRatio() > ft2.first->getFoundRatio();});

    for(const std::pair<MapPoint::Ptr, Vector2d> &it : cell)
    {
        logs_.num_total_project++;
        const MapPoint::Ptr &mpt = it.first;
        Vector2d px = it.second;
        int level = 0;
        int result = reprojectMapPoint(frame, mpt, px, level, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2, verbose_);

        mpt->increaseVisible(result+1);

        if(result != 1)
            continue;

        Feature::Ptr new_ft = Feature::create(Corner(px[0], px[1], -1, level), frame->cam_->lift(px));
        frame->addMapPointFeatureMatch(mpt, new_ft);
        mpt->increaseFound(2);

        return true;
    }

    return false;
}

int FeatureTracker::matchMapPointsFromLastFrame(const Frame::Ptr &frame_cur, const Frame::Ptr &frame_last)
{
    if(frame_last == nullptr || frame_cur == nullptr)
        return 0;

    std::vector<MapPoint::Ptr> mpts = frame_last->getMapPoints();

    int matches_count = 0;
    for(const MapPoint::Ptr &mpt : mpts)
    {
        if(!mpt || mpt->isBad())
            continue;

        const Vector3d mpt_cur = frame_cur->Tcw() * mpt->pose();
        if(mpt_cur[2] < 0.0f)
            continue;

        Vector2d px_cur(frame_cur->cam_->project(mpt_cur));
        if(!frame_cur->cam_->isInFrame(px_cur.cast<int>(), options_.border))
            continue;

        logs_.num_total_project++;

        int level_cur = 0;
        int result = reprojectMapPoint(frame_cur, mpt, px_cur, level_cur, options_.num_align_iter, options_.max_align_epsilon, options_.max_align_error2, verbose_);

        mpt->increaseVisible(result+1);

        if(result != 1)
            continue;

        Feature::Ptr new_ft = Feature::create(px_cur, frame_cur->cam_->lift(px_cur), level_cur);
        frame_cur->addMapPointFeatureMatch(mpt, new_ft);
        mpt->increaseFound(2);

        const size_t id = grid_.getIndex(std::make_pair(mpt, px_cur));
        grid_.setMask(id);

        matches_count++;
    }

    return matches_count;
}

int FeatureTracker::reprojectMapPoint(const Frame::Ptr &frame,
                                      const MapPoint::Ptr &mpt,
                                      Vector2d &px_cur,
                                      int &level_cur,
                                      const int max_iterations,
                                      const double epslion,
                                      const double threshold,
                                      bool verbose)
{
    static const int patch_size = AlignPatch::Size;
    static const int patch_border_size = AlignPatch::SizeWithBorder;
    const int TH_SSD = AlignPatch::Area * threshold;

    KeyFrame::Ptr kf_ref;
    if(!mpt->getCloseViewObs(frame, kf_ref, level_cur))
        return -1;

    const size_t idx = mpt->getFeatureIndex(kf_ref);
    const Feature::Ptr ft_ref = kf_ref->getFeatureByIndex(idx);
    if(!ft_ref)
        return -1;

    const Vector3d obs_ref_dir(kf_ref->pose().translation() - mpt->pose());
    const SE3d T_cur_from_ref = frame->Tcw() * kf_ref->pose();

    Matrix2d A_cur_from_ref;
    utils::getWarpMatrixAffine(kf_ref->cam_, frame->cam_, ft_ref->px_, ft_ref->fn_, ft_ref->corner_.level,
                               obs_ref_dir.norm(), T_cur_from_ref, patch_size, A_cur_from_ref);

    // TODO 如果Affine很小的话，则不用warp
    const cv::Mat image_ref = kf_ref->getImage(ft_ref->corner_.level);
    Matrix<float, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<float, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                ft_ref->px_, ft_ref->corner_.level, level_cur);

    const cv::Mat image_cur = frame->getImage(level_cur);

    const double factor = Frame::scale_factors_.at(level_cur);
    Vector3d estimate(0,0,0); estimate.head<2>() = px_cur / factor;

    static bool show = false;
    if(show)
    {
        cv::Mat show_ref, show_cur;
        cv::cvtColor(image_ref, show_ref, CV_GRAY2RGB);
        cv::cvtColor(image_cur, show_cur, CV_GRAY2RGB);
        cv::circle(show_ref, cv::Point2i((int)ft_ref->px_[0], (int)ft_ref->px_[1])*Frame::inv_scale_factors_.at(ft_ref->corner_.level), 3, cv::Scalar(255,0,0));
        cv::circle(show_cur, cv::Point2i((int)estimate[0], (int)estimate[1]), 3, cv::Scalar(255,0,0));
        cv::imshow("ref track", show_ref);
        cv::imshow("cur track", show_cur);
        cv::waitKey(0);
    }

    bool matched = AlignPatch::align2DI(image_cur, patch_with_border, estimate, max_iterations, epslion, verbose);
    if(!matched)
        return 0;

    ZSSD<float, patch_size> zssd(patch_with_border.block(1,1,8,8));
    Matrix<float, patch_size, patch_size, RowMajor> patch_cur;
    utils::interpolateMat<uchar, float, patch_size>(image_cur, patch_cur, estimate[0], estimate[1]);
    float score = zssd.compute_score(patch_cur);
    if(score > TH_SSD)
        return 0;

    px_cur = estimate.head<2>() * factor;

    return 1;
}

bool FeatureTracker::trackFeature(const Frame::Ptr &frame_ref,
                                  const Frame::Ptr &frame_cur,
                                  const Feature::Ptr &ft_ref,
                                  const MapPoint::Ptr &mpt,
                                  Vector2d &px_cur,
                                  int &level_cur,
                                  const int max_iterations,
                                  const double epslion,
                                  const double threshold,
                                  bool verbose)
{
    static const int patch_size = AlignPatch::Size;
    static const int patch_border_size = AlignPatch::SizeWithBorder;
    const int TH_SSD = AlignPatch::Area * threshold;

    const Vector3d obs_ref_dir(frame_ref->pose().translation() - mpt->pose());
    const SE3d T_cur_from_ref = frame_cur->Tcw() * frame_ref->pose();

    Matrix2d A_cur_from_ref;
    utils::getWarpMatrixAffine(frame_ref->cam_, frame_cur->cam_, ft_ref->px_, ft_ref->fn_, ft_ref->corner_.level,
                               obs_ref_dir.norm(), T_cur_from_ref, patch_size, A_cur_from_ref);

    level_cur = utils::getBestSearchLevel(A_cur_from_ref, frame_cur->nlevels_-1, Frame::scale_factor_);
//    std::cout << "A:\n" << A_cur_from_ref << std::endl;

    const cv::Mat image_ref = frame_ref->getImage(ft_ref->corner_.level);
    Matrix<float, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<float, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                ft_ref->px_, ft_ref->corner_.level, level_cur);

    const cv::Mat image_cur = frame_cur->getImage(level_cur);

    const double factor = Frame::scale_factors_.at(level_cur);
    Vector3d estimate(0,0,0); estimate.head<2>() = px_cur / factor;

    bool matched = AlignPatch::align2DI(image_cur, patch_with_border, estimate, max_iterations, epslion, verbose);
    if(!matched)
        return false;

    ZSSD<float, patch_size> zssd(patch_with_border.block(1,1,8,8));
    Matrix<float, patch_size, patch_size, RowMajor> patch_cur;
    utils::interpolateMat<uchar, float, patch_size>(image_cur, patch_cur, estimate[0], estimate[1]);
    float score = zssd.compute_score(patch_cur);
    if(score > TH_SSD)
        return false;

    px_cur = estimate.head<2>() * factor;

    return true;
}

bool FeatureTracker::findSubpixelFeature(const Frame::Ptr &frame_ref,
                                         const Frame::Ptr &frame_cur,
                                         const Feature::Ptr &ft_ref,
                                         const Vector3d &p3d,
                                         Vector2d &px_cur,
                                         const int level_cur,
                                         const int max_iterations,
                                         const double epslion,
                                         const double threshold,
                                         bool verbose)
{
    static const int patch_size = AlignPatch::Size;
    static const int patch_border_size = AlignPatch::SizeWithBorder;
    const int TH_SSD = AlignPatch::Area * threshold;

    const Vector3d obs_ref_dir(frame_ref->pose().translation() - p3d);
    const SE3d T_cur_from_ref = frame_cur->Tcw() * frame_ref->pose();

    Matrix2d A_cur_from_ref;
    utils::getWarpMatrixAffine(frame_ref->cam_, frame_cur->cam_, ft_ref->px_, ft_ref->fn_, ft_ref->corner_.level,
                               obs_ref_dir.norm(), T_cur_from_ref, patch_size, A_cur_from_ref);

    const cv::Mat image_ref = frame_ref->getImage(ft_ref->corner_.level);
    Matrix<float, patch_border_size, patch_border_size, RowMajor> patch_with_border;
    utils::warpAffine<float, patch_border_size>(image_ref, patch_with_border, A_cur_from_ref,
                                                ft_ref->px_, ft_ref->corner_.level, level_cur);

    const cv::Mat image_cur = frame_cur->getImage(level_cur);

    const double factor = Frame::scale_factors_.at(level_cur);
    Vector3d estimate(0,0,0); estimate.head<2>() = px_cur / factor;

    bool matched = AlignPatch::align2DI(image_cur, patch_with_border, estimate, max_iterations, epslion, verbose);
    if(!matched)
        return false;

    ZSSD<float, patch_size> zssd(patch_with_border.block(1,1,8,8));
    Matrix<float, patch_size, patch_size, RowMajor> patch_cur;
    utils::interpolateMat<uchar, float, patch_size>(image_cur, patch_cur, estimate[0], estimate[1]);
    float score = zssd.compute_score(patch_cur);
    if(score > TH_SSD)
        return false;

    px_cur = estimate.head<2>() * factor;

    return true;
}

int FeatureTracker::searchBoWForTriangulation(const KeyFrame::Ptr &keyframe1,
                                              const KeyFrame::Ptr &keyframe2,
                                              std::map<size_t, size_t> &matches,
                                              int max_desp, double max_epl_err)
{
    const DBoW3::FeatureVector &feat_vec1 = keyframe1->feat_vec_;
    const DBoW3::FeatureVector &feat_vec2 = keyframe2->feat_vec_;

    DBoW3::FeatureVector::const_iterator ft1_itr = feat_vec1.begin();
    DBoW3::FeatureVector::const_iterator ft2_itr = feat_vec2.begin();
    DBoW3::FeatureVector::const_iterator ft1_end = feat_vec1.end();
    DBoW3::FeatureVector::const_iterator ft2_end = feat_vec2.end();

    Matrix3d K1, K2;
    cv::cv2eigen(keyframe1->cam_->K(), K1);
    cv::cv2eigen(keyframe2->cam_->K(), K2);
    const Matrix3d E12 = utils::Fundamental::computeE12(keyframe1->Tcw(), keyframe2->Tcw());
    const Matrix3d F12 = K1.transpose().inverse() * E12 * K2.inverse();
//
//    const SE3d  T12 = keyframe1->Tcw() * keyframe2->Twc();
//    const Matrix3d E12_temp = Sophus::SO3d::hat(T12.translation()) * T12.so3().matrix();
//
//    std::cout << "E: " << E12 << "\n " << E12_temp << std::endl;

    //showMatches(keyframe1, keyframe2);


//    {
//        std::vector<size_t> mptmatches = keyframe1->getMapPointMatchesVec();
//        std::vector<MapPoint::Ptr> mpts = keyframe1->getMapPoints();
//
//        for(const size_t &idx : mptmatches)
//        {
//            MapPoint::Ptr mpt = mpts[idx];
//            const size_t idx2 = mpt->getFeatureIndex(keyframe2);
//            if(idx2 < 0)
//                continue;
//
//            cv::Mat desp1 = keyframe1->descriptors_[idx];
//            cv::Mat desp2 = keyframe2->descriptors_[idx2];
//
//            int dist = DBoW3::DescManip::distance_8uc1(desp1, desp2);
//
//            const Feature::Ptr ft1 = keyframe1->getFeatureByIndex(idx);
////            const cv::Point2d px1(ft1->fn_[0]/ft1->fn_[2], ft1->fn_[0]/ft1->fn_[2]);
//            const cv::Point2d px1(ft1->px_[0], ft1->px_[1]);
//
//            const Feature::Ptr ft2 = keyframe2->getFeatureByIndex(idx2);
////            const cv::Point2d px2(ft2->fn_[0]/ft2->fn_[2], ft2->fn_[0]/ft2->fn_[2]);
//            const cv::Point2d px2(ft2->px_[0], ft2->px_[1]);
//
//            showEplMatch(keyframe1->getImage(0), keyframe2->getImage(0), F12, ft1->px_, ft2->px_);
//
//            double err1, err2;
//            utils::Fundamental::computeErrors(px2, px1, F12, err2, err1);
//
//            double err = utils::Fundamental::computeErrorSquared(keyframe1->pose().translation(), ft1->fn_, T12.inverse(), ft2->fn_.head<2>()/ft2->fn_[2]);
//
//            std::cout << "dist: " << dist << ", " << err1 << ", " << err2  <<", " << err<< std::endl;
//        }
//    }


    const Matrix3d E21 = E12.transpose();

    std::vector<size_t> matched_idx(keyframe2->getFeatures().size(), -1);
    std::vector<int> matched_dist(keyframe2->getFeatures().size(), 256);

    int dist_outlier = 0;
    int epl_outlier = 0;
    while(ft1_itr != ft1_end && ft2_itr != ft2_end)
    {
        if(ft1_itr->first == ft2_itr->first)
        {
            for(const size_t &idx1 : ft1_itr->second)
            {
                MapPoint::Ptr mpt1 = keyframe1->getMapPointByIndex(idx1);
                if(mpt1) continue;

                //! ft1
                const Feature::Ptr ft1 = keyframe1->getFeatureByIndex(idx1);
                const cv::Point2d px1(ft1->fn_[0]/ft1->fn_[2], ft1->fn_[1]/ft1->fn_[2]);
                const cv::Mat &descriptor1 = keyframe1->descriptors_[idx1];

                const double sigma1 = Frame::level_sigma2_.at(ft1->corner_.level);

                int best_dist = max_desp;
                int best_idx2 = -1;
                for(const size_t &idx2 : ft2_itr->second)
                {
                    MapPoint::Ptr mpt2 = keyframe2->getMapPointByIndex(idx2);
                    if(mpt2) continue;

                    //! ft2
                    const Feature::Ptr ft2 = keyframe2->getFeatureByIndex(idx2);
                    const cv::Mat &descriptor2 = keyframe2->descriptors_[idx2];

                    const int dist = DBoW3::DescManip::distance_8uc1(descriptor1, descriptor2);
                    if(dist > best_dist)
                    {
                        dist_outlier++;
                        continue;
                    }

                    // TODO  MORE CHECK

                    const cv::Point2d px2(ft2->fn_[0]/ft2->fn_[2], ft2->fn_[1]/ft2->fn_[2]);
                    const double sigma2 = Frame::level_sigma2_.at(ft2->corner_.level);

                    double err1, err2;
                    utils::Fundamental::computeErrors(px1, px2, E21, err1, err2);

                    if(err1 > max_epl_err*sigma1 || err2 > max_epl_err*sigma2)
                    {
                        epl_outlier++;
                        continue;
                    }

                    if(dist < best_dist)
                    {
                        best_dist = dist;
                        best_idx2 = idx2;
                    }
                }


                if(best_idx2 > 0)
                {
//                    const Feature::Ptr ft2 = keyframe2->getFeatureByIndex(best_idx2);
//                    showEplMatch(keyframe1->getImage(0), keyframe2->getImage(0), F12, ft1->px_, ft2->px_);
                    if(matched_idx[best_idx2] != -1 && matched_dist[best_idx2] < best_dist)
                        continue;

                    if(matched_idx[best_idx2] != -1)
                        matches.erase(matched_idx[best_idx2]);

                    matches.emplace(idx1, best_idx2);

                    matched_idx[best_idx2] = idx1;
                    matched_dist[best_idx2] = best_dist;
                }
            }

            ft1_itr++;
            ft2_itr++;
        }
        else if(ft1_itr->first < ft2_itr->first)
        {
            ft1_itr = feat_vec1.lower_bound(ft2_itr->first);
        }
        else
        {
            ft2_itr = feat_vec2.lower_bound(ft1_itr->first);
        }
    }

//    std::cout << "Outlier: " << dist_outlier << ", " << epl_outlier << std::endl;

    return matches.size();
}

int FeatureTracker::searchBowByProjection(const KeyFrame::Ptr &keyframe,
                                          const std::vector<MapPoint::Ptr> &mpts,
                                          std::map<MapPoint::Ptr, size_t> &matches,
                                          int max_desp,
                                          double threshold)
{
    std::unordered_set<MapPoint::Ptr> mpts_found;
    {
        const std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts = keyframe->getMapPointFeaturesMatched();
        for(const auto &mpt_ft : mpt_fts)
            mpts_found.insert(mpt_ft.first);
    }

    const SE3d Tcw = keyframe->Tcw();
    const Vector3d Owc = keyframe->pose().translation();

    for(const MapPoint::Ptr &mpt : mpts)
    {
        if(mpt->isBad() || mpts_found.count(mpt))
            continue;

        const Vector3d Pw = mpt->pose();
        const Vector3d Pc = Tcw * Pw;

        if(Pc[2] <= 0.0f)
            continue;

        const Vector2d px_mpt = keyframe->cam_->project(Pc);
        if(!keyframe->cam_->isInFrame(px_mpt.cast<int>()))
            continue;

        const double max_dist = mpt->getMaxDistanceInvariance();
        const double min_dist = mpt->getMinDistanceInvariance();

        const Vector3d dist = Owc - Pw;
        const double dist_norm = dist.norm();

        if(dist_norm > max_dist || dist_norm < min_dist)
            continue;

        const Vector3d obs_dir = mpt->getMeanViewDirection();
        const double cos_theta = dist.dot(obs_dir)/dist_norm;

        if(cos_theta < 0.5)
            continue;

        const int level = mpt->predictScale(dist_norm, Frame::nlevels_-1);
        const double radius = threshold * Frame::scale_factors_.at(level);
        const std::vector<size_t> indices = keyframe->getFeaturesInArea(px_mpt[0], px_mpt[1], radius, level-1, level+1);
        if(indices.empty())
            continue;

        const cv::Mat desp1 = mpt->descriptor();

        int best_dist = max_desp;
        int best_idx = -1;
        for(const size_t &idx : indices)
        {
            const Feature::Ptr &ft = keyframe->getFeatureByIndex(idx);
            const Vector2d &px = ft->px_;
            const Vector2d rpj_err = px_mpt - px;
            const double rpj_err_square = rpj_err.squaredNorm();
            const double sigma2 = Frame::level_sigma2_.at(ft->corner_.level);
            if(rpj_err_square > sigma2 * 5.991)
                continue;

            const cv::Mat &desp2 = keyframe->descriptors_.at(idx);
            const int desp_dist = DBoW3::DescManip::distance(desp1, desp2);
            if(desp_dist < best_dist)
            {
                best_dist = desp_dist;
                best_idx = idx;
            }
        }

        if(best_idx >= 0)
            matches.emplace(mpt, best_idx);
    }

    return (int)matches.size();
}


void FeatureTracker::showMatches(const Frame::Ptr frame1, const Frame::Ptr frame2)
{
    const std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts1 = frame1->getMapPointFeaturesMatched();
    const std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts2 = frame2->getMapPointFeaturesMatched();

    const cv::Mat &image1 = frame1->getImage(0);
    const cv::Mat &image2 = frame2->getImage(0);

    const int cols = image1.cols;
    const int rows = image1.rows;
    cv::Mat show(rows, cols*2, CV_8UC1);
    image1.copyTo(show.colRange(0, cols));
    image2.copyTo(show.colRange(cols, cols*2));
    cv::cvtColor(show, show, CV_GRAY2RGB);

    cv::RNG rng(12345);
    for(const auto item1 : mpt_fts1)
    {
        const MapPoint::Ptr &mpt = item1.first;
        const auto iter = mpt_fts2.find(mpt);
        if(iter == mpt_fts2.end())
            continue;

        const Feature::Ptr &ft1 = item1.second;
        const Feature::Ptr &ft2 = iter->second;

        cv::Point2i px1(ft1->corner_.x, ft1->corner_.y);
        cv::Point2i px2(ft2->corner_.x+cols, ft2->corner_.y);

        cv::Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle(show, px1, 3, color);
        cv::circle(show, px2, 3, color);
        cv::line(show, px1, px2, color);
    }

    cv::imwrite("KeyFrame Matches.png", show);
}

void FeatureTracker::showMatches(const Frame::Ptr frame1,
                                 const Frame::Ptr frame2,
                                 const std::vector<std::pair<size_t, size_t>> &matches)
{
    const cv::Mat &image1 = frame1->getImage(0);
    const cv::Mat &image2 = frame2->getImage(0);
    const std::vector<Feature::Ptr> fts1 = frame1->getFeatures();
    const std::vector<Feature::Ptr> fts2 = frame2->getFeatures();
    std::vector<bool> mask1(fts1.size());
    std::vector<bool> mask2(fts2.size());

    const int cols = image1.cols;
    const int rows = image1.rows;
    cv::Mat show(rows, cols*2, CV_8UC1);
    image1.copyTo(show.colRange(0, cols));
    image2.copyTo(show.colRange(cols, cols*2));
    cv::cvtColor(show, show, CV_GRAY2RGB);

    cv::RNG rng(12345);
    for(const auto item : matches)
    {
        const size_t &idx1 = item.first;
        const size_t &idx2 = item.second;

        mask1[idx1] = true;
        mask2[idx2] = true;

        const Feature::Ptr &ft1 = fts1[idx1];
        const Feature::Ptr &ft2 = fts2[idx2];

        cv::Point2i px1(ft1->corner_.x, ft1->corner_.y);
        cv::Point2i px2(ft2->corner_.x+cols, ft2->corner_.y);

        cv::Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle(show, px1, 3, color);
        cv::circle(show, px2, 3, color);
        cv::line(show, px1, px2, color);
    }

    for(size_t i = 0; i < mask1.size(); i++)
    {
        if(mask1[i]) continue;

        const Feature::Ptr &ft1 = fts1[i];
        cv::Point2i px1(ft1->corner_.x, ft1->corner_.y);
        cv::Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle(show, px1, 3, color);
    }

    for(size_t i = 0; i < mask2.size(); i++)
    {
        if(mask2[i]) continue;

        const Feature::Ptr &ft2 = fts2[i];
        cv::Point2i px2(ft2->corner_.x+cols, ft2->corner_.y);
        cv::Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::circle(show, px2, 3, color);
    }

    cv::imwrite("KeyFrame Matches.png", show);
//    cv::imshow("KeyFrame Matches", show);
//    cv::waitKey(0);
}

void FeatureTracker::showEplMatch(const cv::Mat &image1,
                                  const cv::Mat &image2,
                                  const Matrix3d &F12,
                                  const Vector2d &px1,
                                  const Vector2d &px2)
{
    cv::Mat show1 = image1.clone();
    cv::Mat show2 = image2.clone();
    if(show1.channels() == 1)
        cv::cvtColor(show1, show1, CV_GRAY2RGB);
    if(show2.channels() == 1)
        cv::cvtColor(show2, show2, CV_GRAY2RGB);

    const double u1 = px1[0];
    const double v1 = px1[1];
    const double u2 = px2[0];
    const double v2 = px2[1];

    //! X1^T * F12 * X2
    //! epipolar line in the second image L1 = (a1, b1, c1)^T = F12  * X2
    const double a1 = F12(0,0) * u2 + F12(0,1) * v2 + F12(0,2);
    const double b1 = F12(1,0) * u2 + F12(1,1) * v2 + F12(1,2);
    const double c1 = F12(2,0) * u2 + F12(2,1) * v2 + F12(2,2);
    //! epipolar line in the first image  L2 = (a2, b2, c2)^T = X1^T * F12
    const double a2 = u1 * F12(0,0) + v1 * F12(1,0) + F12(2,0);
    const double b2 = u1 * F12(0,1) + v1 * F12(1,1) + F12(2,1);
    const double c2 = u1 * F12(0,2) + v1 * F12(1,2) + F12(2,2);

    const int rows = image1.rows;
    const int cols = image1.cols;

    cv::Point2i pt11, pt12;
    if(std::abs(a1) > std::abs(b1))
    {
        pt11.y = 0;
        pt11.x = -c1/a1;

        pt12.y = rows;
        pt12.x = -(b1*rows+c1)/a1;
    }
    else
    {
        pt11.x = 0;
        pt11.y = -c1/b1;

        pt12.x = cols;
        pt12.y = - (a1*cols+c1)/b1;
    }

    cv::Point2i pt21, pt22;
    if(std::abs(a2) > std::abs(b2))
    {
        pt21.y = 0;
        pt21.x = -c2/a2;

        pt22.y = rows;
        pt22.x = -(b2*rows+c2)/a2;
    }
    else
    {
        pt21.x = 0;
        pt21.y = -c2/b2;

        pt22.x = cols;
        pt22.y = - (a2*cols+c2)/b2;
    }

    cv::Scalar color1(0, 255, 0);
    cv::Scalar color2(0, 0, 255);
    cv::circle(show1, cv::Point(u1,v1), 3, color1);
    cv::circle(show2, cv::Point(u2,v2), 3, color1);

    cv::line(show1, pt11, pt12, color2);
    cv::line(show2, pt21, pt22, color2);

    cv::imwrite("imag1.png", show1);
    cv::imwrite("imag2.png", show2);
}

void FeatureTracker::showAllFeatures(const KeyFrame::Ptr &keyframe)
{
    std::vector<Feature::Ptr> fts = keyframe->getFeatures();
    std::vector<size_t> mpt_indices = keyframe->getMapPointMatchIndices();
    std::vector<size_t> seed_matched_indices = keyframe->getSeedMatchIndices();
    std::vector<size_t> seed_created_indices = keyframe->getSeedCreateIndices();

    cv::Mat img = keyframe->getImage(0);
    cv::Mat show;
    cv::cvtColor(img, show, CV_GRAY2RGB);

    cv::Scalar color0(150, 100, 0);
    cv::Scalar color1(0, 255, 0);
    cv::Scalar color2(0, 0, 255);
    cv::Scalar color3(255, 0, 255);

    std::unordered_map<size_t, size_t> masked_fts;
    for(const size_t &idx : mpt_indices)
        masked_fts.emplace(idx, 1);

    for(const size_t &idx : seed_matched_indices)
        masked_fts.emplace(idx, 2);

    for(const size_t &idx : seed_created_indices)
        masked_fts.emplace(idx, 3);

    for(size_t idx = 0; idx < fts.size(); idx++)
    {
        Feature::Ptr &ft = fts[idx];
        cv::Point2i px(std::round(ft->px_[0]), std::round(ft->px_[1]));

        const auto itr = masked_fts.find(idx);
        if(itr == masked_fts.end())
            cv::circle(show, px, 3, color0);
        else if(itr->second == 1)
            cv::circle(show, px, 3, color1);
        else if(itr->second == 2)
            cv::circle(show, px, 3, color2);
        else if(itr->second == 3)
            cv::circle(show, px, 3, color3);

    }

//    cv::imwrite("ft_imag.png", show);
//    cv::imshow("KeyFrame feature", show);
//    cv::waitKey(0);
}

}