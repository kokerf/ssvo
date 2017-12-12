#ifndef _SSVO_ALIGNMENT_HPP_
#define _SSVO_ALIGNMENT_HPP_

#include "global.hpp"
#include "config.hpp"
#include "keyframe.hpp"
#include "frame.hpp"

namespace ssvo {

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

    Sophus::SE3d T_cur_from_ref_;
};


class Align2DI : public Align<4, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Align2DI(bool verbose=false):verbose_(verbose){}

    bool run(const Matrix<uchar, Dynamic, Dynamic, RowMajor> &image,
             const Matrix<double, PatchArea, 1> &patch,
             const Matrix<double, PatchArea, 1> &patch_gx,
             const Matrix<double, PatchArea, 1> &patch_gy,
             Eigen::Vector3d &estimate, const int max_iterations = 30, const double epslion = 1E-2f);

private:

    const bool verbose_;

    const int border_ = HalfPatchSize+1;

    Matrix<double, Parameters, Parameters, RowMajor> invHessian_;
    Eigen::Vector3d estimate_;
};

}

#endif //_SSVO_ALIGNMENT_HPP_
