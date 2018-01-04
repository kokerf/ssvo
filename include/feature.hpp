#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "global.hpp"

namespace ssvo {

class MapPoint;
class KeyFrame;

class Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Feature> Ptr;

    Vector2d px;
    Vector3d fn;
    int level;
    std::shared_ptr<MapPoint> mpt;

    inline static Ptr create(const Vector2d &_px, const Vector3d &_fn, int _level, const std::shared_ptr<MapPoint> &_mpt)
    {return std::make_shared<Feature>(Feature(_px, _fn, _level, _mpt));}

private:
    Feature(const Vector2d &px, const Vector3d &fn, int _level, const std::shared_ptr<MapPoint> &mpt);

};

typedef std::list<Feature::Ptr> Features;

//! modified from SVO, https://github.com/uzh-rpg/rpg_svo/blob/master/svo/include/svo/depth_filter.h#L35
/// A seed is a probabilistic depth estimate for a single pixel.
struct Seed
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Seed> Ptr;

    const std::shared_ptr<KeyFrame> kf;     //!< Reference KeyFrame, where the seed created.
    const Vector3d fn_ref;                  //!< Pixel in the keyframe's normalized plane where the depth should be computed.
    const Vector2d px_ref;                  //!< Pixel matched in current frame
    const int level_ref;                    //!< Corner detected level in refrence frame
    Vector2d px_cur;                        //!< Pixel matched in current frame
    int level_cur;                          //!< Corner detected level in current frame
    double a;                               //!< a of Beta distribution: When high, probability of inlier is large.
    double b;                               //!< b of Beta distribution: When high, probability of outlier is large.
    double mu;                              //!< Mean of normal distribution.
    double z_range;                         //!< Max range of the possible depth.
    double sigma2;                          //!< Variance of normal distribution.
    Matrix2d patch_cov;                     //!< Patch covariance in reference image.

    std::list<std::pair<double, double> > history;

    double computeTau(const SE3d &T_ref_cur, const Vector3d& f, const double z, const double px_error_angle);
//    double computeVar(const SE3d &T_cur_ref, const double d, const double delta);
    void update(const double x, const double tau2);
    inline static Ptr create(const std::shared_ptr<KeyFrame> &kf, const Vector2d &px, const Vector3d &fn, const int level, double depth_mean, double depth_min)
    {return std::make_shared<Seed>(Seed(kf, px, fn, level, depth_mean, depth_min));}

private:
    Seed(const std::shared_ptr<KeyFrame> &kf, const Vector2d &px, const Vector3d &fn, const int level, double depth_mean, double depth_min);
};

typedef std::list<Seed::Ptr> Seeds;

}

#endif