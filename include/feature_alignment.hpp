#ifndef _SSVO_FEATURE_ALIGNMENT_HPP_
#define _SSVO_FEATURE_ALIGNMENT_HPP_

#include "global.hpp"
#include "config.hpp"
#include "keyframe.hpp"
#include "frame.hpp"
#include "utils.hpp"
#include "pattern.hpp"

namespace ssvo {

//! ====================== Patch align
template <int PatchSize>
class AlignPatch{
public:
    enum {
        Size = PatchSize,
        Area = Size*Size,
        HalfSize = Size/2,
        SizeWithBorder = Size+2,
    };

    static bool align2DI(const cv::Mat &image_cur,
                         const Matrix<float, SizeWithBorder, SizeWithBorder, RowMajor> &patch_ref_with_border,
                         Vector3d &estimate,
                         const int max_iterations = 30,
                         const double epslion = 1E-2f,
                         const bool verbose = false);

    static bool align2DI(const cv::Mat &image_cur,
                         const Matrix<float, Area, 1> &patch_ref,
                         const Matrix<float, Area, 1> &patch_ref_gx,
                         const Matrix<float, Area, 1> &patch_ref_gy,
                         Vector3d &estimate,
                         const int max_iterations = 30,
                         const double epslion = 1E-2f,
                         const bool verbose = false);
};

typedef AlignPatch<8> AlignPatch8x8;
typedef AlignPatch<16> AlignPatch16x16;
typedef AlignPatch<32> AlignPatch32x32;


//! ====================== Pattern align
class AlignPattern
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum {
        Num = 32,
        Size = 8,
        HalfSize = Size/2,
        SizeWithBorder = Size+2,
    };

    static const Pattern<float, Num, Size> pattern_;

    static bool align2DI(const cv::Mat &image_cur,
                         const Matrix<float, Num, 1> &patch_ref,
                         const Matrix<float, Num, 1> &patch_ref_gx,
                         const Matrix<float, Num, 1> &patch_ref_gy,
                         Vector3d &estimate,
                         const int max_iterations = 30,
                         const double epslion = 1E-2f,
                         const bool verbose = false);
};


template <typename T, int Size>
class ZSSD{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ZSSD(const Matrix<T, Size, Size, RowMajor> &patch_ref):
        A(patch_ref)
    {
        A.array() -= A.mean();
    }

    T compute_score(const Matrix<T, Size, Size, RowMajor> &patch_cur)
    {
        Matrix<T, Size, Size, RowMajor> B = patch_cur.array() - patch_cur.mean();

        return (A-B).norm();
    }

    T threshold() const {return threshold_;}

private:
    Matrix<T, Size, Size, RowMajor> A;
    const T threshold_ = Size * 500;
};

}

#endif //_SSVO_FEATURE_ALIGNMENT_HPP_
