#ifndef _SSVO_IMAGEALIGNMENT_HPP_
#define _SSVO_IMAGE_ALIGNMENT_HPP_

#include "global.hpp"
#include "config.hpp"
#include "keyframe.hpp"
#include "frame.hpp"
#include "utils.hpp"
#include "pattern.hpp"

namespace ssvo {

void calculateLightAffine(const cv::Mat &I, const cv::Mat &J, float &a, float &b);

template <int Size, int Num>
class Align{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum {
        PatchSize = Size,
        PatchArea = PatchSize*PatchSize,
        HalfPatchSize = Size/2,
        Parameters = Num,
    };

    std::list<std::string> logs_;

protected:
    Matrix<double, Dynamic, PatchArea, RowMajor> ref_patch_cache_;
    Matrix<double, Dynamic, Num, RowMajor> jacbian_cache_;
    Matrix<double, Num, Num, RowMajor> Hessian_;
    Matrix<double, Num, 1> Jres_;
};


class AlignSE3: public Align<4, 6>
{
public:

    AlignSE3(bool verbose=false, bool visible=false);

    int run(Frame::Ptr reference_frame, Frame::Ptr current_frame,
            int top_level, int bottom_level, double scale_factor, int max_iterations = 30, double epslion = 1E-5f);

private:

    int computeReferencePatches(int level, const std::vector<Feature::Ptr> &fts, const std::vector<MapPoint::Ptr> &mpts);

    double computeResidual(int level, int N);

private:

    const bool verbose_;
    const bool visible_;

    Frame::Ptr ref_frame_;
    Frame::Ptr cur_frame_;

    int count_;
    std::vector<bool> visiable_fts_;
    Matrix<double, 3, Dynamic, RowMajor> ref_feature_cache_;

    SE3d T_cur_from_ref_;
};


//! ========================== Utils =========================================
namespace utils{

int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level, const float scale_factor);

void getWarpMatrixAffine(const AbstractCamera::Ptr &cam_ref,
                         const AbstractCamera::Ptr &cam_cur,
                         const Vector2d &px_ref,
                         const Vector3d &f_ref,
                         const int level_ref,
                         const double depth_ref,
                         const SE3d &T_cur_ref,
                         const int patch_size,
                         Matrix2d &A_cur_ref);

template<typename Td, int size>
void warpAffine(const cv::Mat &img_ref,
                Matrix<Td, size, size, RowMajor> &patch,
                const Matrix2d &A_cur_from_ref,
                const Vector2d &px_ref,
                const int level_ref,
                const int level_cur)
{
    assert(img_ref.type() == CV_8UC1);

    const Matrix2f A_ref_from_cur = A_cur_from_ref.inverse().cast<float>();
    if(std::isnan(A_ref_from_cur(0,0)))
    {
        LOG(ERROR) << "Affine warp is Nan";
        return;
    }

    const Vector2f px_ref_pyr = px_ref.cast<float>() / Frame::scale_factors_.at(level_ref);
    const float half_patch_size = size * 0.5;
    const float px_pyr_scale = Frame::scale_factors_.at(level_cur);
    for(int y = 0; y < size; ++y)
    {
        for(int x = 0; x < size; ++x)
        {
            Vector2f px_patch(x-half_patch_size, y-half_patch_size);
            px_patch *= px_pyr_scale;//! A_ref_from_cur is for level-0, so transform to it
            Vector2f affine = A_ref_from_cur*px_patch;
            const Vector2f px(affine + px_ref_pyr);

            if(px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
                patch(y, x) = 0;
            else
                patch(y, x) = utils::interpolateMat<uchar, Td>(img_ref, px[0], px[1]);
        }
    }
}

}

}


#endif //SSVO_IMAGE_ALIGNMENT_H
