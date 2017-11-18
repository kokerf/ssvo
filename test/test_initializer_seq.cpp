#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "feature_detector.hpp"
#include "initializer.hpp"
#include "config.hpp"
#include "bundle_adjustment.hpp"
#ifdef WIN32
#include <io.h>
#else
#include <dirent.h>
#endif

using std::string;

#ifdef WIN32
void loadImages(const string &file_directory, std::vector<string> &image_filenames)
{
    _finddata_t file;
    intptr_t fileHandle;
    std::string filename = file_directory + "\\*.jpg";
    fileHandle = _findfirst(filename.c_str(), &file);
    if (fileHandle  != -1L) {
        do {
            std::string image_name = file_directory + "\\" + file.name;
            image_filenames.push_back(image_name);
            std::cout << image_name << std::endl;
        } while (_findnext(fileHandle, &file) == 0);
    }

    _findclose(fileHandle);
    std::sort(image_filenames.begin(), image_filenames.end());
}
#else
void loadImages(const std::string &strFileDirectory, std::vector<string> &vstrImageFilenames)
{
    DIR* dir = opendir(strFileDirectory.c_str());
    dirent* p = NULL;
    while((p = readdir(dir)) != NULL)
    {
        if(p->d_name[0] != '.')
        {
            std::string imageFilename = strFileDirectory + string(p->d_name);
            vstrImageFilenames.push_back(imageFilename);
            //cout << imageFilename << endl;
        }
    }

    closedir(dir);
    std::sort(vstrImageFilenames.begin(),vstrImageFilenames.end());
}
#endif

void evalueErrors(ssvo::KeyFrame::Ptr kf1, ssvo::KeyFrame::Ptr kf2, double& error)
{
    const int N = kf1->fts_.size();
    double residuals[2] = {0,0};
    Quaterniond Q = kf2->getRotation();
    Vector3d t = kf2->getTranslation();
    for(int i = 0; i < N; i++)
    {
        ssvo::Feature::Ptr ft1 = kf1->fts_[i];

        ssvo::MapPoint::Ptr mpt = ft1->mpt;

        if(mpt == nullptr)
            continue;

        ssvo::Feature::Ptr ft2 = mpt->findObservation(kf2);

        Vector3d p1 = mpt->getPose();
        Vector3d p2 = Q._transformVector(p1) + t;

        double predicted_x1 = p1[0] / p1[2];
        double predicted_y1 = p1[1] / p1[2];
        double dx1 = predicted_x1 - ft1->ft[0];
        double dy1 = predicted_y1 - ft1->ft[1];
        residuals[0] += dx1*dx1 + dy1*dy1;

        double predicted_x2 = p2[0] / p2[2];
        double predicted_y2 = p2[1] / p2[2];
        double dx2 = predicted_x2 - ft2->ft[0];
        double dy2 = predicted_y2 - ft2->ft[1];
        residuals[1] += dx2*dx2 + dy2*dy2;
    }

    error = 0.5*(residuals[0] + residuals[1]);
}

std::string ssvo::Config::FileName;

int main(int argc, char const *argv[])
{
    google::InitGoogleLogging(argv[0]);

    if (argc != 3) {
        std::cout << "Usge: ./test_initializer path_to_sequence configflie" << std::endl;
        return -1;
    }

    ssvo::Config::FileName = std::string(argv[2]);
    int fps = ssvo::Config::cameraFps();
    int width = ssvo::Config::imageWidth();
    int height = ssvo::Config::imageHeight();
    int image_border = ssvo::Config::imageBorder();
    int grid_size = ssvo::Config::gridSize();
    int grid_min_size = ssvo::Config::gridMinSize();
    int fast_max_threshold = ssvo::Config::fastMaxThreshold();
    int fast_min_threshold = ssvo::Config::fastMinThreshold();
    double fast_min_eigen = ssvo::Config::fastMinEigen();

    std::string dir_name = argv[1];
    std::vector<string> img_file_names;
    loadImages(dir_name, img_file_names);

    cv::Mat K = ssvo::Config::cameraIntrinsic();
    cv::Mat DistCoef = ssvo::Config::cameraDistCoef();

    ssvo::Camera::Ptr camera = ssvo::Camera::create(ssvo::Config::imageWidth(), ssvo::Config::imageHeight(), K, DistCoef);
    ssvo::FastDetector fast(width, height, image_border, 3, 100, grid_size, grid_min_size);

    ssvo::Initializer initializer(K, DistCoef);

    cv::Mat cur_img;
    cv::Mat ref_img;
    int initial = 0;
    std::vector<ssvo::Corner> corners;
    std::vector<ssvo::Corner> old_corners;
    std::vector<cv::Point2f> pts, upts;
    std::vector<cv::Point2d> fts;
    for(std::vector<std::string>::iterator i = img_file_names.begin(); i != img_file_names.end(); ++i)
    {
        cv::Mat img = cv::imread(*i, CV_LOAD_IMAGE_UNCHANGED);
        if(img.empty()) throw std::runtime_error("Could not open image: " + *i);

        cv::cvtColor(img, cur_img, cv::COLOR_RGB2GRAY);

        if(initial == 0)
        {
            ref_img = cur_img.clone();
            ImgPyr img_pyr;
            ssvo::createPyramid(cur_img, img_pyr);

            fast.detect(img_pyr, corners, old_corners, fast_max_threshold, fast_min_threshold, fast_min_eigen);
            std::for_each(corners.begin(), corners.end(), [&](ssvo::Corner& corner){pts.push_back(cv::Point2f(corner.x, corner.y));});
            cv::undistortPoints(pts, upts, K, DistCoef);
            fts.clear();fts.reserve(corners.size());
            std::for_each(upts.begin(), upts.end(), [&](cv::Point2f& pt){fts.push_back(cv::Point2d((double)pt.x, (double)pt.y));});
            LOG(INFO) << "All corners: " << pts.size();
            int succeed = initializer.addFirstImage(cur_img, pts, fts);
            if(succeed == ssvo::SUCCESS) initial = 1;
        }
        else if(initial == 1)
        {
            ssvo::InitResult result = initializer.addSecondImage(cur_img);
            if(result == ssvo::RESET) {
                initial = 0;
                continue;
            }
            else if(result == ssvo::SUCCESS)
                break;

            std::vector<cv::Point2f> pts_ref, pts_cur;
            initializer.getTrackedPoints(pts_ref, pts_cur);

            cv::Mat match_img = img.clone();
            for(size_t i=0; i<pts_ref.size();i++)
            {
                cv::line(match_img, pts_ref[i], pts_cur[i],cv::Scalar(0,0,70));
            }

            cv::imshow("KeyPoints detectByImage", match_img);
        }

        cv::waitKey(fps);
    }

    ssvo::Frame::Ptr frame1 = ssvo::Frame::create(ref_img, 0, camera);
    ssvo::Frame::Ptr frame2 = ssvo::Frame::create(cur_img, 0, camera);
    std::vector<cv::Point2f> pts1, pts2;
    std::vector<cv::Point2d> fts1, fts2;
    std::vector<Vector3d> p3ds;
    cv::Mat inliers;
    MatrixXd T;
    initializer.getResults(pts1, pts2, fts1, fts2, p3ds, inliers, T);

    std::vector<cv::Point2d>::iterator fts1_iter = fts1.begin();
    std::vector<cv::Point2f>::iterator pts1_iter = pts1.begin();
    std::vector<cv::Point2d>::iterator fts2_iter = fts2.begin();
    std::vector<cv::Point2f>::iterator pts2_iter = pts2.begin();
    std::vector<Vector3d>::iterator p3ds_iter = p3ds.begin();

    const uchar* inliers_ptr = inliers.ptr<uchar>(0);
    for(size_t j = 0; p3ds_iter != p3ds.end() ; ++j, fts1_iter++, fts2_iter++, pts1_iter++, pts2_iter++, p3ds_iter++)
    {
        if(!inliers_ptr[j])
        {
            continue;
        }
        Vector3d ft1(fts1_iter->x, fts1_iter->y, 1);
        Vector3d ft2(fts2_iter->x, fts2_iter->y, 1);
        Vector2d px1(pts1_iter->x, pts1_iter->y);
        Vector2d px2(pts2_iter->x, pts2_iter->y);

        ssvo::MapPoint::Ptr mpt = ssvo::MapPoint::create(*p3ds_iter);
        frame1->fts_.push_back(ssvo::Feature::create(px1, ft1,0,mpt));
        frame2->fts_.push_back(ssvo::Feature::create(px2, ft2,0,mpt));
    }
    frame1->setRotation(Quaterniond::Identity());
    frame1->setTranslation(0,0,0);
    frame2->setPose(T);

    ssvo::KeyFrame::Ptr keyframe1 = ssvo::KeyFrame::create(frame1);
    ssvo::KeyFrame::Ptr keyframe2 = ssvo::KeyFrame::create(frame2);
    keyframe1->updateObservation();
    keyframe2->updateObservation();

    double error = 0;
    evalueErrors(keyframe1, keyframe2, error);
    LOG(INFO) << "Error before BA: " << error;

    ssvo::BA::twoViewBA(keyframe1, keyframe2, nullptr);

    evalueErrors(keyframe1, keyframe2, error);
    LOG(INFO) << "Error after BA: " << error;


    cv::waitKey(0);

    return 0;
}