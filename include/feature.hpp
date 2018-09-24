#ifndef _FEATURE_HPP_
#define _FEATURE_HPP_

#include <vector>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

#include "global.hpp"

namespace ssvo {

struct Corner{
    float x;        //!< x-coordinate of corner in the image.
    float y;        //!< y-coordinate of corner in the image.
    int level;      //!< pyramid level of the corner.
    float score;    //!< shi-tomasi score of the corner.
    //float angle;  //!< for gradient-features: dominant gradient angle.
    Corner() : x(-1), y(-1), level(-1), score(-1) {}
    Corner(int x, int y, float score, int level) : x(x), y(y), level(level), score(score) {}

    Corner(const Corner& other): x(other.x), y(other.y), level(other.level), score(other.score) {}
};

typedef std::vector<Corner> Corners;

class Feature
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<Feature> Ptr;

    Corner corner_;

    Vector2d px_;

    Vector3d fn_;

    inline static Ptr create(const Corner &corner)
    {return std::make_shared<Feature>(Feature(corner));}

    inline static Ptr create(const Vector2d &px, const int level = 0)
    {return std::make_shared<Feature>(Feature(px));}

private:

    Feature(const Corner &corner):
        corner_(corner), px_(Vector2d(corner.x, corner.y))
    {}

    Feature(const Vector2d &px, const int level = 0):
        corner_(px[0], px[1], -1, level), px_(px)
    {}

};

typedef std::list<Feature::Ptr> Features;

}

#endif