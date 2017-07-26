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

void evalueErrors(std::vector<cv::Point2d>& fts1, std::vector<cv::Point2d>& fts2, std::vector<Vector3d>& p3ds, Quaterniond& Q, Vector3d& t, double& error)
{
    const int N = p3ds.size();
    double residuals[2] = {0,0};
    for(int i = 0; i < N; i++)
    {
        cv::Point2d &ft1 = fts1[i];
        cv::Point2d &ft2 = fts2[i];
        Vector3d &p1 = p3ds[i];
        Vector3d p2 = Q._transformVector(p1) + t;

        double predicted_x1 = p1[0] / p1[2];
        double predicted_y1 = p1[1] / p1[2];
        double dx1 = predicted_x1 - ft1.x;
        double dy1 = predicted_y1 - ft1.y;
        residuals[0] += dx1*dx1 + dy1*dy1;

        double predicted_x2 = p2[0] / p2[2];
        double predicted_y2 = p2[1] / p2[2];
        double dx2 = predicted_x2 - ft2.x;
        double dy2 = predicted_y2 - ft2.y;
        residuals[1] += dx2*dx2 + dy2*dy2;
    }

    error = 0.5*(residuals[0] + residuals[1]);
}

std::string ssvo::Config::FileName;

int main(int argc, char const *argv[])
{
    if (argc != 2) {
        std::cout << "Usge: ./test_initializer path_to_sequence configflie" << std::endl;
    }

    ssvo::Config::FileName = std::string(argv[2]);

    std::string dir_name = argv[1];
    std::vector<string> img_file_names;
    loadImages(dir_name, img_file_names);

    cv::Mat K = ssvo::Config::cameraIntrinsic();
    cv::Mat DistCoef = ssvo::Config::cameraDistCoef();

    ssvo::Camera camera(ssvo::Config::imageWidth(), ssvo::Config::imageHeight(), K, DistCoef);
    ssvo::FastDetector fast_detector(300, 0, true);
    ssvo::Initializer initializer(K, DistCoef);

    int fps = ssvo::Config::cameraFps();
    cv::Mat cur_img;
    cv::Mat ref_img;
    int initial = 0;
    std::vector<cv::KeyPoint> kps;
    std::vector<cv::KeyPoint> kps_old;
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
            ssvo::Frame::createPyramid(cur_img, img_pyr);
            fast_detector.detectByImage(img_pyr, kps, kps_old);
            cv::KeyPoint::convert(kps, pts);
            cv::undistortPoints(pts, upts, K, DistCoef);
            fts.clear();fts.reserve(kps.size());
            std::for_each(upts.begin(), upts.end(), [&](cv::Point2f& pt){fts.push_back(cv::Point2d((double)pt.x, (double)pt.y));});
            std::cout << "All corners: " << pts.size() <<std::endl;
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
            for(int i=0; i<pts_ref.size();i++)
            {
                cv::line(match_img, pts_ref[i], pts_cur[i],cv::Scalar(0,0,70));
            }

            cv::imshow("KeyPoints detectByImage", match_img);
        }

        cv::waitKey(fps);
    }

    std::vector<cv::Point2f> pts1, pts2;
    std::vector<cv::Point2d> fts1, fts2;
    std::vector<Vector3d> p3ds;
    cv::Mat inliers;
    MatrixXd T;
    initializer.getResults(pts1, pts2, fts1, fts2, p3ds, inliers, T);

    std::vector<cv::Point2d>::iterator fts1_iter = fts1.begin();
    std::vector<cv::Point2d>::iterator fts2_iter = fts2.begin();
    std::vector<Vector3d>::iterator p3ds_iter = p3ds.begin();

    const uchar* inliers_ptr = inliers.ptr<uchar>(0);
    for(int j = 0; p3ds_iter != p3ds.end() ; ++j)
    {
        if(!inliers_ptr[j])
        {
            fts1_iter = fts1.erase(fts1_iter);
            fts2_iter = fts2.erase(fts2_iter);
            p3ds_iter = p3ds.erase(p3ds_iter);
            continue;
        }

        fts1_iter++;
        fts2_iter++;
        p3ds_iter++;
    }

    Matrix3d R = T.block(0,0,3,3);
    Vector3d t = T.block(0,3,3,1);
    Quaterniond Q2(R);

    double error = 0;
    evalueErrors(fts1, fts2, p3ds, Q2, t, error);
    std::cout << "Error before BA: " << error << std::endl;


    //! full BA
    ceres::Problem problem;
    ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization();
    double Q1_arr[4] = {1,0,0,0};
    double Q2_arr[4] = {Q2.w(), Q2.x(), Q2.y(), Q2.z()};
    double t1_arr[3] = {0,0,0};
    double t2_arr[3] = {t[0], t[1], t[2]};

    problem.AddParameterBlock(Q1_arr, 4, local_parameterization);
    problem.AddParameterBlock(Q2_arr, 4, local_parameterization);
    problem.AddParameterBlock(t1_arr, 3);
    problem.AddParameterBlock(t2_arr, 3);
    problem.SetParameterBlockConstant(Q1_arr);
    problem.SetParameterBlockConstant(t1_arr);
    for(int id = 0; id < p3ds.size();id++)
    {
        cv::Point2d &ft1 = fts1[id];
        cv::Point2d &ft2 = fts2[id];

        ceres::CostFunction* cost_function2 = ssvo::ReprojectionError::Create(ft2.x, ft2.y);
        problem.AddResidualBlock(cost_function2, NULL, Q2_arr, t2_arr, p3ds[id].data());

        ceres::CostFunction* cost_function1 = ssvo::ReprojectionError::Create(ft1.x, ft1.y);
        problem.AddResidualBlock(cost_function1, NULL, Q1_arr, t1_arr, p3ds[id].data());

    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Quaterniond Q2_(Q2_arr[0], Q2_arr[1], Q2_arr[2], Q2_arr[3]);
    Vector3d t_(t2_arr);
    evalueErrors(fts1, fts2, p3ds, Q2_, t_, error);
    std::cout << "Error after BA: " << error << std::endl;

    std::cout << summary.FullReport() << "\n";

    cv::waitKey(0);

    return 0;
}