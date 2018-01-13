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
    pangolin::Var<bool> menu_follow_camera("menu.Follow Camera", true, true);
    pangolin::Var<bool> menu_show_connections("menu.Connections", true, true);
    pangolin::Var<bool> menu_show_current_connections("menu.Connections_cur", true, true);


    bool following_camera = true;
    bool show_connections = true;
    bool show_current_connections = true;

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(WIN_WIDTH, WIN_HEIGHT, 500, 500, WIN_WIDTH/2, WIN_HEIGHT/2, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -1, -0.5, 0, 0, 0, 0, -1, 0));

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

        if(menu_follow_camera)
        {
            pangolin::OpenGlMatrix pose;
            getCurrentCameraPose(pose);
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, -1, -0.5, 0, 0, 0, 0, -1, 0));
            s_cam.Follow(pose);
            following_camera = true;
        }
        else if(!menu_follow_camera && following_camera)
        {
            following_camera = false;
        }

        if(menu_show_connections)
        {
            show_connections = true;
        }
        else if(!menu_show_connections && show_connections)
        {
            show_connections = false;
        }

        if(menu_show_current_connections)
        {
            show_current_connections = true;
        }
        else if(!menu_show_current_connections && show_current_connections)
        {
            show_current_connections = false;
        }

        pangolin::glDrawAxis(0.1);
        drawMapPoints();

        drawKeyFrames(show_connections, show_current_connections);

        drawCurrentFrame();

        drawCurrentImage(imageTexture);

        image_viewer.Activate();
        glColor3f(1.0,1.0,1.0);
        imageTexture.RenderToViewportFlipY();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    pangolin::DestroyWindow(win_name);
}

void Viewer::setCurrentFrame(const Frame::Ptr &frame)
{
    std::lock_guard<std::mutex> lock(mutex_frame_);
    frame_ = frame;
    camera_pose_ = frame_->pose().matrix();
    cv::cvtColor(frame_->getImage(0), image_, CV_GRAY2RGB);
}

void Viewer::getCurrentCameraPose(pangolin::OpenGlMatrix &M)
{
    Eigen::Map<Matrix<pangolin::GLprecision, 4, 4> > T(M.m);
    {
        std::lock_guard<std::mutex> lock(mutex_frame_);
        T = camera_pose_;
    }
}

void Viewer::drawKeyFrames(bool show_connections, bool show_current)
{
    std::vector<KeyFrame::Ptr> kfs = map_->getAllKeyFrames();

    std::set<KeyFrame::Ptr> loacl_kfs;
    if(show_current)
    {
        const KeyFrame::Ptr &ref_kf = frame_->getRefKeyFrame();
        if(ref_kf != nullptr)
        {
            loacl_kfs = ref_kf->getConnectedKeyFrames();
            loacl_kfs.insert(ref_kf);
        }
    }

    for(const KeyFrame::Ptr &kf : kfs)
    {
        SE3d pose = kf->pose();
        if(loacl_kfs.count(kf))
            drawCamera(pose.matrix(), cv::Scalar(0.0, 0.5, 1.0));
        else
            drawCamera(pose.matrix(), cv::Scalar(0.0, 1.0, 0.2));
    }

    if(!show_connections)
        return;

    glLineWidth(key_frame_graph_line_width);
    glColor4f(0.0f,1.0f,0.0f,0.6f);
    glBegin(GL_LINES);

    for(const KeyFrame::Ptr &kf : kfs)
    {
        Vector3f O1 = kf->pose().translation().cast<float>();
        const std::set<KeyFrame::Ptr> conect_kfs = kf->getConnectedKeyFrames();
        for(const KeyFrame::Ptr &ckf : conect_kfs)
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

void Viewer::drawCurrentFrame()
{
    std::lock_guard<std::mutex> lock(mutex_frame_);
    drawCamera(camera_pose_.matrix(), cv::Scalar(0.0, 0.0, 1.0));
}

void Viewer::drawCurrentImage(pangolin::GlTexture &gl_texture)
{
    std::lock_guard<std::mutex> lock(mutex_frame_);
    if(image_.empty())
        return;

    gl_texture.Upload(image_.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void Viewer::drawMapPoints()
{
    std::unordered_map<MapPoint::Ptr, Feature::Ptr> obs_mpts;
    {
        std::lock_guard<std::mutex> lock(mutex_frame_);
        obs_mpts = frame_->features();
    }

    std::vector<MapPoint::Ptr> mpts = map_->getAllMapPoints();

    glPointSize(map_point_size);
    glBegin(GL_POINTS);
    for(const MapPoint::Ptr &mpt : mpts)
    {
        Vector3d pose = mpt->pose();
        if(obs_mpts.count(mpt))
            glColor3f(1.0,0.0,0.3);
        else
            glColor3f(0.0,0.0,1.0);
//        float rate = (float)mpt->getFoundRatio();
//        glColor3f((1-rate)*rate, 0, rate*rate);
        glVertex3f(pose[0], pose[1], pose[2]);
    }
    glEnd();
}

void Viewer::drawCamera(const Matrix4d &pose, cv::Scalar color)
{
    const float w = key_frame_size ;
    const float h = key_frame_size * 0.57f;
    const float z = key_frame_size * 0.6f;

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
