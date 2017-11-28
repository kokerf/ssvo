#ifndef _SSVO_ALIGNMENT_HPP_
#define _SSVO_ALIGNMENT_HPP_

#include "global.hpp"
#include "config.hpp"
#include "keyframe.hpp"
#include "frame.hpp"

namespace ssvo {

template <int T>
struct Align{
    enum {
        PatchSize = T,
        PatchArea = PatchSize*PatchSize,
        HalfPatchSize = T/2,
    };
};


class AlignSE3: public Align<4>
{
public:

    AlignSE3(const int max_iterations, const double epslion);

    bool run(Frame::Ptr reference_frame, Frame::Ptr current_frame);

private:

    int computeReferencePatches(int level);

    double computeResidual(int level, int N);

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    const int top_level_;

    const int max_iterations_;

    const double epslion_squared_;

private:

    Frame::Ptr ref_frame_;
    Frame::Ptr cur_frame_;

    std::vector<bool> visiable_fts_;
    Matrix<double, 3, Dynamic, RowMajor> ref_feature_cache_;
    Matrix<double, Dynamic, Dynamic, RowMajor> ref_patch_cache_;
    Matrix<double, Dynamic, 6, RowMajor> jacbian_cache_;
    Matrix<double, 6, 6, RowMajor> Hessian_;
    Matrix<double, 6, 1> Jres_;

    Sophus::SE3d T_cur_from_ref_;
};

}

#endif //_SSVO_ALIGNMENT_HPP_
