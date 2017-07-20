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

    static float cameraFps(){return getInstance().fps;}

    static int initMinCorners(){return getInstance().init_min_corners;}

    static int initMinTracked(){return getInstance().init_min_tracked;}

    static int initMinDisparity(){return getInstance().init_min_disparity;}

    static int initMinInliers(){return getInstance().init_min_inliers;}

    static float initSigma(){return getInstance().init_sigma;}

    static float initUnSigma(){return getInstance().init_unsigma;}

    static int initMaxRansacIters(){return getInstance().init_max_iters;}

    static int imageBorder(){return getInstance().image_border;}

    static int gridMinSize(){return getInstance().grid_min_size;}

    static int gridMaxFeatures(){return getInstance().grid_max_fts;}

    static int fastMaxThreshold(){return getInstance().fast_max_threshold;}

    static int fastMinThreshold(){return getInstance().fast_min_threshold;}

    static float fastMinEigen(){return getInstance().fast_min_eigen;}

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
        fx = (float)fs["Camera.fx"];
        fy = (float)fs["Camera.fy"];
        cx = (float)fs["Camera.cx"];
        cy = (float)fs["Camera.cy"];

        k1 = (float)fs["Camera.k1"];
        k2 = (float)fs["Camera.k2"];
        p1 = (float)fs["Camera.p1"];
        p2 = (float)fs["Camera.p2"];

        K = cv::Mat::eye(3,3,CV_32F);
        K.at<float>(0,0) = fx;
        K.at<float>(1,1) = fy;
        K.at<float>(0,2) = cx;
        K.at<float>(1,2) = cy;

        DistCoef = cv::Mat::zeros(4,1,CV_32F);
        DistCoef.at<float>(0) = k1;
        DistCoef.at<float>(1) = k2;
        DistCoef.at<float>(2) = p1;
        DistCoef.at<float>(3) = p2;

        width = (int)fs["Camera.width"];
        height = (int)fs["Camera.height"];
        fps = (float)fs["Camera.fps"];

        //! initializer parameters
        init_min_corners = (int)fs["Initializer.min_corners"];
        init_min_tracked = (int)fs["Initializer.min_tracked"];
        init_min_disparity = (int)fs["Initializer.min_disparity"];
        init_min_inliers = (int)fs["Initializer.min_inliers"];
        init_sigma = (float)fs["Initializer.sigma"];
        init_unsigma = init_sigma / MIN(fx,fy);
        init_max_iters = (int)fs["Initializer.ransac_max_iters"];

        //! FAST detector parameters
        image_border = (int)fs["FastDetector.image_border"];
        grid_min_size = (int)fs["FastDetector.grid_min_size"];
        grid_max_fts = (int)fs["FastDetector.grid_max_fts"];
        fast_max_threshold = (int)fs["FastDetector.fast_max_threshold"];
        fast_min_threshold = (int)fs["FastDetector.fast_min_threshold"];
        fast_min_eigen = (float)fs["FastDetector.fast_min_eigen"];

        fs.release();
    }

public:
    //! config file's name
    static string FileName;

private:
    //! camera parameters
    float fx, fy, cx, cy;
    float k1, k2, p1, p2;
    cv::Mat K;
    cv::Mat DistCoef;
    int width;
    int height;
    float fps;

    //! initializer parameters
    int init_min_corners;
    int init_min_tracked;
    int init_min_disparity;
    int init_min_inliers;
    float init_sigma;
    float init_unsigma;
    int init_max_iters;

    //! FAST detector parameters
    int image_border;
    int grid_min_size;
    int grid_max_fts;
    int fast_max_threshold;
    int fast_min_threshold;
    float fast_min_eigen;

};

}

#endif