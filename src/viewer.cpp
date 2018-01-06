#include <pangolin/pangolin.h>
#include "viewer.hpp"

namespace ssvo{

Viewer::Viewer(const Map::Ptr &map, cv::Size image_size) : map_(map), image_size_(image_size)
{
    camera_pose_.setIdentity();

    map_point_size = 3;
    key_frame_size = 0.05;
    key_frame_line_width= 2;
    key_frame_graph_line_width = 1;

    pongolin_thread_ = std::make_shared<std::thread>(std::bind(&Viewer::run, this));
}

void Viewer::run()
{
    const int WIN_WIDTH = 1280;
    const int WIN_HEIGHT = 720;
    const int UI_WIDTH = 160;
    const int UI_HEIGHT = 160;
    const int IMAGE_WIDTH = UI_HEIGHT *image_size_.width/image_size_.height;
    const int IMAGE_HEIGHT = UI_HEIGHT;

    const string win_name = "SSVO Viewer";
    pangolin::CreateWindowAndBind(win_name, WIN_WIDTH, WIN_HEIGHT);
    glEnable(GL_DEPTH_TEST);


    glEnable(GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::CreatePanel("menu").SetBounds(pangolin::Attach::Pix(WIN_HEIGHT-UI_HEIGHT), 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));
    pangolin::Var<bool> menu_follow_camera("menu.Follow Camera",true, true);


    bool following_camera = true;

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(WIN_WIDTH, WIN_HEIGHT, 500, 500, WIN_WIDTH/2, WIN_HEIGHT/2, 0.1, 1000),
        pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisNegY));

    // Create Interactive View in window
    pangolin::View& camera_viewer = pangolin::Display("Camera")
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, (-1.0*WIN_WIDTH)/WIN_HEIGHT)
        .SetHandler(new pangolin::Handler3D(s_cam));

    pangolin::View& image_viewer = pangolin::Display("Image")
        .SetBounds(pangolin::Attach::Pix(WIN_HEIGHT-IMAGE_HEIGHT), 1, pangolin::Attach::Pix(UI_WIDTH), pangolin::Attach::Pix(UI_WIDTH+IMAGE_WIDTH), (-1.0*image_size_.width/image_size_.height))
        .SetLock(pangolin::LockLeft, pangolin::LockTop);

    pangolin::GlTexture imageTexture(image_size_.width, image_size_.height, GL_RGB, false, 0, GL_BGR,GL_UNSIGNED_BYTE);

    while(!pangolin::ShouldQuit())
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        camera_viewer.Activate(s_cam);

        glClearColor(1.0f,1.0f,1.0f,1.0f);
//
//        if(menu_follow_camera)
//        {
//            std::lock_guard<std::mutex> lock(mutex_camera_);
//            s_cam.Follow(camera_pose_);
//            following_camera = true;
//        }
//        else if(!menu_follow_camera && following_camera)
//        {
//            following_camera = false;
//        }

//        glPointSize(10.0f);
//        glBegin(GL_POINTS);
//        glColor3f(1.0,0.0,0.0);
//        glVertex3f(0.0f,0.0f,0.0f);
//        glVertex3f(0,0,1);
//        glEnd();

        //pangolin::glDrawLine(0.0,0.0,0.0,0,0,0.5);
        pangolin::glDrawAxis(0.1);
        drawMapPoints();

        drawKeyFrames();

        drawCurFrame();

        cv::Mat im;
        {
            std::lock_guard<std::mutex> lock(mutex_image_);
            if(!image_.empty())
                imageTexture.Upload(image_.data, GL_RGB, GL_UNSIGNED_BYTE);
//            cv::imshow("image_",image_);
//            cv::waitKey(1);
        }

        image_viewer.Activate();
        glColor3f(1.0,1.0,1.0);
        imageTexture.RenderToViewportFlipY();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    pangolin::DestroyWindow(win_name);
}

void Viewer::showImage(const cv::Mat &image)
{
    std::lock_guard<std::mutex> lock(mutex_image_);
    LOG_ASSERT(!image.empty());
    if(image.channels() != 3)
        cv::cvtColor(image, image_, CV_GRAY2RGB);
    else
        image_ = image.clone();
}

void Viewer::setCurrentCameraPose(const Matrix4d &pose)
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    camera_pose_ = pose;
}

void Viewer::drawKeyFrames()
{
    std::vector<KeyFrame::Ptr> kfs = map_->getAllKeyFrames();

    for(KeyFrame::Ptr kf : kfs)
    {
        SE3d pose = kf->pose();
        drawCamera(pose.matrix(), cv::Scalar(0.0, 1.0, 0.2));
    }

    glLineWidth(key_frame_graph_line_width);
    glColor4f(0.0f,1.0f,0.0f,0.6f);
    glBegin(GL_LINES);

    for(KeyFrame::Ptr kf : kfs)
    {
        Vector3f O1 = kf->pose().translation().cast<float>();
        const std::set<KeyFrame::Ptr> conect_kfs = kf->getConnectedKeyFrames();
        for(KeyFrame::Ptr ckf : conect_kfs)
        {
            if(ckf->id_ < kf->id_)
                continue;

            Vector3f O2 = ckf->pose().translation().cast<float>();
            glVertex3f(O1[0], O1[1], O1[2]);
            glVertex3f(O2[0], O2[1], O2[2]);
        }

    }
    glEnd();

}

void Viewer::drawCurFrame()
{
    std::lock_guard<std::mutex> lock(mutex_pose_);
    drawCamera(camera_pose_.matrix(), cv::Scalar(0.0, 0.0, 1.0));
}

void Viewer::drawMapPoints()
{
    std::vector<MapPoint::Ptr> mpts = map_->getAllMapPoints();

    glPointSize(map_point_size);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,1.0);
    for(MapPoint::Ptr mpt : mpts)
    {
        Vector3d pose = mpt->pose();
        glVertex3f(pose[0], pose[1], pose[2]);
    }
    glEnd();
}

void Viewer::drawCamera(const Matrix4d &pose, cv::Scalar color)
{
    const float w = key_frame_size ;
    const float h = key_frame_size * 0.57;
    const float z = key_frame_size * 0.6;

    glPushMatrix();

    //! col major
    glMultMatrixd(pose.data());

    glLineWidth(key_frame_line_width);
    glColor3f((GLfloat)color[0], (GLfloat)color[1], (GLfloat)color[2]);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

}
