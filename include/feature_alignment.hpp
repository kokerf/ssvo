#ifndef _SSVO_FEATURE_ALIGNMENT_HPP_
#define _SSVO_FEATURE_ALIGNMENT_HPP_

#include "global.hpp"
#include "config.hpp"
#include "keyframe.hpp"
#include "frame.hpp"
#include "utils.hpp"
#include "pattern.hpp"

namespace ssvo {

class Align2DI
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum {
        PatchSize = 8,
        PatchArea = PatchSize*PatchSize,
        HalfPatchSize = PatchSize/2,
    };

    Align2DI(bool verbose=false):verbose_(verbose){}

    bool run(const cv::Mat &image_cur,
             const Matrix<float, PatchSize+2, PatchSize+2, RowMajor> &patch_ref_with_border,
             Vector3d &estimate, const int max_iterations = 30, const double epslion = 1E-2f);

    std::list<std::string> logs_;

private:

    const bool verbose_;
};

//! ====================== Pattern align
template <int Size, int NumPn, int NumPr>
class AlignPattern
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum {
        PatternNum = NumPn,
        PatchSize = Size+2, //! patch size with border
        HalfPatchSize = PatchSize/2,
        ParaNum = NumPr,
    };

    static const Pattern<float, NumPn, Size> pattern_;
    std::list<std::string> logs_;

protected:
    Matrix<float, NumPn, NumPr, RowMajor> jacbian_cache_;
    Matrix<float, NumPr, NumPr, RowMajor> Hessian_;
    Matrix<float, NumPr, NumPr, RowMajor> invHessian_;
    Matrix<float, NumPr, 1> Jres_;
    Matrix<float, NumPr, 1> estimate_;
};


class AlignP2DI : public AlignPattern<13, 49, 3>
{
public:

    AlignP2DI(bool verbose=false) : verbose_(verbose) {}

    bool run(const Matrix<uchar, Dynamic, Dynamic, RowMajor> &image,
             const Matrix<float, PatternNum, 3> &patch_idxy,
             Matrix<double, ParaNum, 1> &estimate, const int max_iterations = 30, const double epslion = 1E-2f);

private:

    const bool verbose_;
    const int border_ = HalfPatchSize;

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
