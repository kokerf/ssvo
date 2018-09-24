#include "viewer.hpp"

namespace ssvo{

Viewer::Viewer(const Map::Ptr &map, cv::Size image_size) :
    map_(map), image_size_(image_size), required_stop_(false), is_finished_(false)
{
    map_point_size = 3;
    key_frame_size = 0.05;
    key_frame_line_width= 2;
    key_frame_graph_line_width = 1;

    pongolin_thread_ = std::make_shared<std::thread>(std::bind(&Viewer::run, this));
}


void Viewer::setStop()
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    required_stop_ = true;
}

bool Viewer::isRequiredStop()
{
    std::lock_guard<std::mutex> lock(mutex_stop_);
    return required_stop_;
}

bool Viewer::waitForFinish()
{
    if(!isRequiredStop())
        setStop();

    if(pongolin_thread_->joinable())
        pongolin_thread_->join();
    
    return true;
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
    pangolin::Var<bool> menu_show_trajectory("menu.Show Trajectory", true, true);
    pangolin::Var<bool> menu_show_keyframe("menu.Show KeyFrame", true, true);
    pangolin::Var<bool> menu_show_connections("menu.Connections", true, true);
    pangolin::Var<bool> menu_show_current_connections("menu.Connections_cur", true, true);

    const int trajectory_duration_max = 10000;
    pangolin::Var<int> settings_trajectory_duration("menu.Traj Duration",1000, 1, trajectory_duration_max,false);


    bool following_camera = true;
    bool show_trajectory = true;
    bool show_keyframe = true;
    bool show_connections = true;
    bool show_current_connections = true;
    int trajectory_duration = -1;

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

    while(!pangolin::ShouldQuit() && !isRequiredStop())
    {
        Frame::Ptr frame;
        cv::Mat image;
        {
            std::lock_guard<std::mutex> lock(mutex_frame_);
            frame = frame_;
            image = image_;
        }

        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        camera_viewer.Activate(s_cam);

        glClearColor(1.0f,1.0f,1.0f,1.0f);


        //! update
        following_camera = menu_follow_camera.Get();
        show_trajectory = menu_show_trajectory.Get();
        show_keyframe = menu_show_keyframe.Get();
        show_connections = menu_show_connections.Get();
        show_current_connections = menu_show_current_connections.Get();

        trajectory_duration = settings_trajectory_duration.Get();
        if(trajectory_duration == trajectory_duration_max) trajectory_duration = -1;

        if(following_camera)
        {
            pangolin::OpenGlMatrix camera_pose;
            Eigen::Map<Matrix<pangolin::GLprecision, 4, 4> > T(camera_pose.m);
            if(frame)
            {
                T = frame->pose().matrix();
            }
            else{
                T.setIdentity();
            }

            //s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0, -1, -0.5, 0, 0, 0, 0, -1, 0));
            s_cam.Follow(camera_pose);
            following_camera = true;
        }

        pangolin::glDrawAxis(0.1);

        drawMapPoints(map_, frame);

        if(show_keyframe)
        {
            KeyFrame::Ptr reference = frame ? frame->getRefKeyFrame() : nullptr;
            drawKeyFrames(map_, reference, show_connections, show_current_connections);
        }

        if(show_trajectory)
            drawTrajectory(trajectory_duration);

        if(frame)
            drawCurrentFrame(frame->pose().matrix(), cv::Scalar(0.0, 0.0, 1.0));


        if(image.empty() && frame != nullptr)
            drawTrackedPoints(frame, image);

        drawCurrentImage(imageTexture, image);

        image_viewer.Activate();
        glColor3f(1.0,1.0,1.0);
        imageTexture.RenderToViewportFlipY();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    pangolin::DestroyWindow(win_name);
}

void Viewer::setCurrentFrame(const Frame::Ptr &frame, const cv::Mat image)
{
    std::lock_guard<std::mutex> lock(mutex_frame_);
    frame_ = frame;
    image_ = image;
    if(frame_)
        frame_trajectory_.push_back(frame_->pose().translation());
}

void Viewer::drawKeyFrames(Map::Ptr &map, KeyFrame::Ptr &reference, bool show_connections, bool show_current)
{
    std::vector<KeyFrame::Ptr> kfs = map->getAllKeyFrames();

    std::set<KeyFrame::Ptr> loacl_kfs;
    if(show_current)
    {
        if(reference != nullptr)
        {
            loacl_kfs = reference->getConnectedKeyFrames();
            loacl_kfs.insert(reference);
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

void Viewer::drawCurrentFrame(const Matrix4d &pose, cv::Scalar color)
{
    drawCamera(pose.matrix(), color);
}

void Viewer::drawCurrentImage(pangolin::GlTexture &gl_texture, cv::Mat &image)
{
    if(image.empty())
        return;

    if(image.type() == CV_8UC1)
        cv::cvtColor(image, image, CV_GRAY2RGB);
    gl_texture.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
    cv::imshow("SSVO Current Image", image);
    cv::waitKey(1);
}

void Viewer::drawMapPoints(Map::Ptr &map, Frame::Ptr &frame)
{
    std::unordered_map<MapPoint::Ptr, Feature::Ptr> obs_mpts;
    if(frame)
        obs_mpts = frame->getMapPointFeatureMatches();

    std::vector<MapPoint::Ptr> mpts = map->getAllMapPoints();

    glPointSize(map_point_size);
    glBegin(GL_POINTS);
    for(const MapPoint::Ptr &mpt : mpts)
    {
        Vector3d pose = mpt->pose();
        if(obs_mpts.count(mpt))
            glColor3f(1.0,0.0,0.3);
//        else if(mpt->observations() == 1)
//             glColor3f(0.0,0.0,0.0);
        else
            glColor3f(0.5,0.5,0.5);
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

void Viewer::drawTrackedPoints(const Frame::Ptr &frame, cv::Mat &dst)
{
    //! draw features
    const cv::Mat src = frame->getImage(0);
    std::unordered_map<MapPoint::Ptr, Feature::Ptr> mpt_fts = frame->getMapPointFeatureMatches();
    cv::cvtColor(src, dst, CV_GRAY2RGB);
    int font_face = 1;
    double font_scale = 0.5;
    for(const auto &mpt_ft : mpt_fts)
    {
        const MapPoint::Ptr &mpt = mpt_ft.first;
        const Feature::Ptr &ft = mpt_ft.second;
        Vector2d ft_px = ft->px_;
        cv::Point2f px(ft_px[0], ft_px[1]);
        cv::Scalar color(0, 255, 0);
        cv::circle(dst, px, 2, color, -1);

        string id_str = std::to_string((frame->Tcw()*mpt->pose()).norm());//ft->mpt_->getFoundRatio());//
        cv::putText(dst, id_str, px-cv::Point2f(1,1), font_face, font_scale, color);
    }

//    //! draw seeds
//    std::vector<Feature::Ptr> seed_fts = frame->getSeeds();
//    for(const Feature::Ptr &ft : seed_fts)
//    {
//        Seed::Ptr seed = ft->seed_;
//        cv::Point2f px(ft->px_[0], ft->px_[1]);
//        double convergence = 0;
//        double scale = MIN(convergence, 256.0) / 256.0;
//        cv::Scalar color(255*scale, 0, 255*(1-scale));
//        cv::circle(dst, px, 2, color, -1);
//
////        string id_str = std::to_string();
////        cv::putText(dst, id_str, px-cv::Point2f(1,1), font_face, font_scale, color);
//    }

}

void Viewer::drawTrajectory(int frame_num)
{
    std::lock_guard<std::mutex> lock(mutex_frame_);
    float color[3] = {1,0,0};
    glColor3f(color[0],color[1],color[2]);
    glLineWidth(2);

    glBegin(GL_LINE_STRIP);

    size_t frame_count_max = frame_num == -1 ? frame_trajectory_.size() : static_cast<size_t>(frame_num);
    size_t frame_count = 0;
    for(auto itr = frame_trajectory_.rbegin(); itr != frame_trajectory_.rend() && frame_count < frame_count_max; itr++, frame_count++)
    {
        glVertex3f((float)(*itr)[0], (float)(*itr)[1], (float)(*itr)[2]);
    }
    glEnd();
}

}
