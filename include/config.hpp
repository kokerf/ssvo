#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include <iostream>
#include <string>

#include <opencv2/core.hpp>

namespace ssvo{

using std::string;

class Config
{
public:

    static cv::Mat cameraIntrinsic(){return getInstance().K;}

    static cv::Mat cameraDistCoef(){return getInstance().DistCoef;}

    static int imageWidth(){return getInstance().width;}

    static int imageHeight(){return getInstance().height;}

    static double cameraFps(){return getInstance().fps;}

    static int initMinCorners(){return getInstance().init_min_corners;}

    static int initMinTracked(){return getInstance().init_min_tracked;}

    static int initMinDisparity(){return getInstance().init_min_disparity;}

    static int initMinInliers(){return getInstance().init_min_inliers;}

    static double initSigma(){return getInstance().init_sigma;}

    static double initSigma2(){return getInstance().init_sigma2;}

    static double initUnSigma(){return getInstance().init_unsigma;}

    static double initUnSigma2(){return getInstance().init_unsigma2;}

    static int initMaxRansacIters(){return getInstance().init_max_iters;}

    static int imageBorder(){return getInstance().image_border;}

    static int gridMinSize(){return getInstance().grid_min_size;}

    static int gridMaxFeatures(){return getInstance().grid_max_fts;}

    static int fastMaxThreshold(){return getInstance().fast_max_threshold;}

    static int fastMinThreshold(){return getInstance().fast_min_threshold;}

    static double fastMinEigen(){return getInstance().fast_min_eigen;}

private:
    static Config& getInstance()
    {
        static Config instance(FileName);
        return instance;
    }

    Config(string& file_name)
    {
        cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
        if(!fs.isOpened())
        {
           std::cerr << "Failed to open settings file at: " << file_name << std::endl;
           exit(-1);
        }

        //! camera parameters
        fx = (double)fs["Camera.fx"];
        fy = (double)fs["Camera.fy"];
        cx = (double)fs["Camera.cx"];
        cy = (double)fs["Camera.cy"];

        k1 = (double)fs["Camera.k1"];
        k2 = (double)fs["Camera.k2"];
        p1 = (double)fs["Camera.p1"];
        p2 = (double)fs["Camera.p2"];

        K = cv::Mat::eye(3,3,CV_64F);
        K.at<double>(0,0) = fx;
        K.at<double>(1,1) = fy;
        K.at<double>(0,2) = cx;
        K.at<double>(1,2) = cy;

        DistCoef = cv::Mat::zeros(4,1,CV_64F);
        DistCoef.at<double>(0) = k1;
        DistCoef.at<double>(1) = k2;
        DistCoef.at<double>(2) = p1;
        DistCoef.at<double>(3) = p2;

        width = (int)fs["Camera.width"];
        height = (int)fs["Camera.height"];
        fps = (double)fs["Camera.fps"];

        //! initializer parameters
        init_min_corners = (int)fs["Initializer.min_corners"];
        init_min_tracked = (int)fs["Initializer.min_tracked"];
        init_min_disparity = (int)fs["Initializer.min_disparity"];
        init_min_inliers = (int)fs["Initializer.min_inliers"];
        init_sigma = (double)fs["Initializer.sigma"];
        init_sigma2 = init_sigma*init_sigma;
        init_unsigma2 = init_sigma2 * 2.0 / (fx*fx+fy*fy);
        init_unsigma = sqrt(init_unsigma2);
        init_max_iters = (int)fs["Initializer.ransac_max_iters"];

        //! FAST detector parameters
        image_border = (int)fs["FastDetector.image_border"];
        grid_min_size = (int)fs["FastDetector.grid_min_size"];
        grid_max_fts = (int)fs["FastDetector.grid_max_fts"];
        fast_max_threshold = (int)fs["FastDetector.fast_max_threshold"];
        fast_min_threshold = (int)fs["FastDetector.fast_min_threshold"];
        fast_min_eigen = (double)fs["FastDetector.fast_min_eigen"];

        fs.release();
    }

public:
    //! config file's name
    static string FileName;

private:
    //! camera parameters
    double fx, fy, cx, cy;
    double k1, k2, p1, p2;
    cv::Mat K;
    cv::Mat DistCoef;
    int width;
    int height;
    double fps;

    //! initializer parameters
    int init_min_corners;
    int init_min_tracked;
    int init_min_disparity;
    int init_min_inliers;
    double init_sigma;
    double init_sigma2;
    double init_unsigma;
    double init_unsigma2;
    int init_max_iters;

    //! FAST detector parameters
    int image_border;
    int grid_min_size;
    int grid_max_fts;
    int fast_max_threshold;
    int fast_min_threshold;
    double fast_min_eigen;

};

}

#endif