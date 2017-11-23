#ifndef _SSVO_VIEWER_HPP_
#define _SSVO_VIEWER_HPP_

#include "global.hpp"
#include "config.hpp"
#include "map.hpp"

namespace ssvo {

class Viewer : public noncopyable
{
public:
    typedef std::shared_ptr<Viewer> Ptr;

    void run();

    void setCurrentCameraPose(const Matrix4d &pose);

    void showImage(const cv::Mat &image);

    static Viewer::Ptr create(Map::Ptr map){ return Viewer::Ptr(new Viewer(map));}

private:

    Viewer(Map::Ptr map);

    void drawMapPoints();

    void drawCamera(const Matrix4d &pose);

    void drawKeyFrames();

private:

    std::shared_ptr<std::thread> thread_;

    Map::Ptr map_;

    cv::Mat image_;
    cv::Size image_size_;
    Matrix4d camera_pose_;

    std::mutex mutex_image_;
    std::mutex mutex_camera_;

    float map_point_size;
    float key_frame_size;
    float key_frame_line_size;

};

}

#endif //_SSVO_VIEWER_HPP_
