#ifndef _SSVO_VIEWER_HPP_
#define _SSVO_VIEWER_HPP_

#include <pangolin/pangolin.h>
#include "global.hpp"
#include "config.hpp"
#include "map.hpp"

namespace ssvo {

class Viewer : public noncopyable
{
public:
    typedef std::shared_ptr<Viewer> Ptr;

    void run();

    void setStop();

    bool waitForFinish();

    void setCurrentFrame(const Frame::Ptr &frame, const cv::Mat image = cv::Mat());

    static Viewer::Ptr create(const Map::Ptr &map, cv::Size image_size){ return Viewer::Ptr(new Viewer(map, image_size));}

private:

    Viewer(const Map::Ptr &map, cv::Size image_size);

    bool isRequiredStop();

    void setFinished();

    void drawMapPoints(Map::Ptr &map, Frame::Ptr &frame);

    void drawCamera(const Matrix4d &pose, cv::Scalar color);

    void drawKeyFrames(Map::Ptr &map, KeyFrame::Ptr &reference, bool show_connections=false, bool show_current=false);

    void drawCurrentFrame(const Matrix4d &pose, cv::Scalar color);

    void drawCurrentImage(pangolin::GlTexture& gl_texture, cv::Mat &image);

    void drawTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst);

    void drawTrajectory(int frame_num = -1);

private:

    std::shared_ptr<std::thread> pongolin_thread_;

    Map::Ptr map_;

    Frame::Ptr frame_;
    cv::Mat image_;
    cv::Size image_size_;

    std::list<Vector3d, aligned_allocator<Vector3d> > frame_trajectory_;

    float map_point_size;
    float key_frame_size;
    float key_frame_line_width;
    float key_frame_graph_line_width;

    bool required_stop_;
    bool is_finished_;

    std::mutex mutex_frame_;
    std::mutex mutex_stop_;
};

}

#endif //_SSVO_VIEWER_HPP_
