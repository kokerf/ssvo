#ifndef _SSVO_ALIGNMENT_HPP_
#define _SSVO_ALIGNMENT_HPP_

#include "global.hpp"
#include "config.hpp"
#include "keyframe.hpp"
#include "frame.hpp"
#include "utils.hpp"

namespace ssvo {

template <int T>
struct Pattern
{
    enum{Size = T};
    const std::array<std::array<int, 2>, T> data;

    inline std::array<int, T> make_index(int stride)
    {
        std::array<int, T> index;
        for(int i=0; i < T; i++){ index[i] = data[i][0] + stride*data[i][1];}
        return index;
    };
};

const Pattern<16> pattern{
    {{
         {0, -3}, {1, -3}, {2, -2}, {3, -1},
         {3, 0}, {3, 1}, {2, 2}, {1, 3},
         {0, 3}, {-1, 3}, {-2, 2}, {-3, 1},
         {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}
     }}
};

const Pattern<8> pattern1{
    {{
        {0, -3}, { 2, -2}, { 3, 0}, { 2,  2},
        {0,  3}, {-2,  2}, {-3, 0}, {-2, -2},
    }}
};


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

    bool run(Frame::Ptr reference_frame, Frame::Ptr current_frame,
             int top_level, int max_iterations = 30, double epslion = 1E-5f);

private:

    int computeReferencePatches(int level);

    double computeResidual(int level, int N);

private:

    const bool verbose_;
    const bool visible_;

    Frame::Ptr ref_frame_;
    Frame::Ptr cur_frame_;

    std::vector<bool> visiable_fts_;
    Matrix<double, 3, Dynamic, RowMajor> ref_feature_cache_;

    SE3d T_cur_from_ref_;
};


class Align2DI : public Align<8, 3>
{
public:

    Align2DI(bool verbose=false):verbose_(verbose){}

    bool run(const Matrix<uchar, Dynamic, Dynamic, RowMajor> &image,
             const Matrix<double, PatchArea, 1> &patch,
             const Matrix<double, PatchArea, 1> &patch_gx,
             const Matrix<double, PatchArea, 1> &patch_gy,
             Vector3d &estimate, const int max_iterations = 30, const double epslion = 1E-2f);

private:

    const bool verbose_;

    const int border_ = HalfPatchSize+1;

    Matrix<double, Parameters, Parameters, RowMajor> invHessian_;
    Vector3d estimate_;
};

template <typename T, int Size>
class ZSSD{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ZSSD(const Matrix<T, Size, 1> &patch_ref):
        A(patch_ref)
    {
        A.array() -= A.mean();
    }

    T compute_score(const Matrix<T, Size, 1> &patch_cur)
    {
        Matrix<T, Size, 1> B = patch_cur.array() - patch_cur.mean();

        return (A-B).norm();
    }

    T threshold() const {return threshold_;}

private:
    Matrix<T, Size, 1> A;
    const T threshold_ = Size * 500;
};


//! ========================== Utils =========================================
namespace utils{

int getBestSearchLevel(const Matrix2d& A_cur_ref, const int max_level);

void getWarpMatrixAffine(const Camera::Ptr &cam_ref,
                         const Camera::Ptr &cam_cur,
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
            Vector2d affine = A_ref_from_cur*px_patch;
            const Vector2d px(affine + px_ref_pyr);

            if(px[0]<0 || px[1]<0 || px[0]>=img_ref.cols-1 || px[1]>=img_ref.rows-1)
                patch(y, x) = 0;
            else
                patch(y, x) = utils::interpolateMat<uchar, Td>(img_ref, px[0], px[1]);
        }
    }
}

}

}

#endif //_SSVO_ALIGNMENT_HPP_
