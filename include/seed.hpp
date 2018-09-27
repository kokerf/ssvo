#ifndef _SSVO_SEED_HPP_
#define _SSVO_SEED_HPP_

#include "global.hpp"

namespace ssvo{

class KeyFrame;

//! modified from SVO, https://github.com/uzh-rpg/rpg_svo/blob/master/svo/include/svo/depth_filter.h#L35
/// A seed is a probabilistic depth estimate for a single pixel.
class Seed
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Seed> Ptr;

    static uint64_t next_id;
    const uint64_t id;
    const std::shared_ptr<KeyFrame> kf;     //!< Reference KeyFrame, where the seed created.
    const size_t ft_idx;                    //!< Correspond feature index in keyframe.
    const Vector3d fn_ref;                  //!< Pixel in the keyframe's normalized plane where the depth should be computed.
    const Vector2d px_ref;                  //!< Pixel matched in reference frame
    const int level_ref;                    //!< Corner detected level in refrence frame

    Vector3d optimal_pose_;

    const static double convergence_rate;

    std::list<std::pair<double, double> > history;

    double computeTau(const SE3d &T_ref_cur, const Vector3d& f, const double z, const double px_error_angle);
    double computeVar(const SE3d &T_cur_ref, const double z, const double delta);
    void update(const double x, const double tau2);
    bool checkConvergence();
    double getInvDepth();
    double getVariance();
    double getInfoWeight();

    inline static Ptr create(const std::shared_ptr<KeyFrame> &kf, const size_t &idx, double depth_mean, double depth_min)
    {return Ptr(new Seed(kf, idx, depth_mean, depth_min));}

private:
    double a;                               //!< a of Beta distribution: When high, probability of inlier is large.
    double b;                               //!< b of Beta distribution: When high, probability of outlier is large.
    double mu;                              //!< Mean of normal distribution.
    double z_range;                         //!< Max range of the possible depth.
    double sigma2;                          //!< Variance of normal distribution.
    Matrix2d patch_cov;                     //!< Patch covariance in reference image.

    std::mutex mutex_seed_;

    Seed(const std::shared_ptr<KeyFrame> &kf, const size_t &idx, double depth_mean, double depth_min);
};

typedef std::list<Seed::Ptr> Seeds;
}

#endif //_SSVO_SEED_HPP_
