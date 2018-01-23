#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_

#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <glog/logging.h>

namespace ssvo{

using std::string;

class Config
{
public:

    static cv::Mat cameraIntrinsic(){return getInstance().K;}

    static cv::Mat cameraDistCoef(){return getInstance().DistCoef;}

    static int imageWidth(){return getInstance().width;}

    static int imageHeight(){return getInstance().height;}

    static int imageTopLevel(){return getInstance().top_level;}

    static double cameraFps(){return getInstance().fps;}

    static double unitPlanePixelLength() { return  getInstance().unit_plane_pixel_length; }

    static int initMinCorners(){return getInstance().init_min_corners;}

    static int initMinTracked(){return getInstance().init_min_tracked;}

    static int initMinDisparity(){return getInstance().init_min_disparity;}

    static int initMinInliers(){return getInstance().init_min_inliers;}

    static double pixelSigma(){return getInstance().init_sigma;}

    static double pixelSigma2(){return getInstance().init_sigma2;}

    static double pixelUnSigma(){return getInstance().init_unsigma;}

    static double pixelUnSigma2(){return getInstance().init_unsigma2;}

    static int initMaxRansacIters(){return getInstance().init_max_iters;}

//    static int imageBorder(){return getInstance().image_border;}

    static int gridSize(){return getInstance().grid_size;}

    static int gridMinSize(){return getInstance().grid_min_size;}

    //static int gridMaxFeatures(){return getInstance().grid_max_fts;}

    static int fastMaxThreshold(){return getInstance().fast_max_threshold;}

    static int fastMinThreshold(){return getInstance().fast_min_threshold;}

    static double fastMinEigen(){return getInstance().fast_min_eigen;}

    static double mapScale(){return getInstance().mapping_scale;}

    static int minConnectionObservations(){return getInstance().mapping_min_connection_observations;}

    static int minCornersPerKeyFrame(){return getInstance().mapping_min_corners;}

    static int alignTopLevel(){return getInstance().align_top_level;}

    static int alignPatchSize(){return getInstance().align_patch_size;}

    static int minQualityFts(){return getInstance().min_quality_fts;}

    static int maxQualityDropFts(){return getInstance().max_quality_drop_fts;}

private:
    static Config& getInstance()
    {
        static Config instance(FileName);
        return instance;
    }

    Config(string& file_name)
    {
        cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
        LOG_ASSERT(fs.isOpened()) << "Failed to open settings file at: " << file_name;

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

        width = (int)fs["Image.width"];
        height = (int)fs["Image.height"];
        top_level = (int)fs["Image.pyramid_levels"];
        fps = (double)fs["Camera.fps"];
        unit_plane_pixel_length = 2.0 / (fx*fx+fy*fy);

        //! FAST detector parameters
//        image_border = (int)fs["FastDetector.image_border"];
        grid_size = (int)fs["FastDetector.grid_size"];
        grid_min_size = (int)fs["FastDetector.grid_min_size"];
        //grid_max_fts = (int)fs["FastDetector.grid_max_fts"];
        fast_max_threshold = (int)fs["FastDetector.fast_max_threshold"];
        fast_min_threshold = (int)fs["FastDetector.fast_min_threshold"];
        fast_min_eigen = (double)fs["FastDetector.fast_min_eigen"];

        //! initializer parameters
        init_min_corners = (int)fs["Initializer.min_corners"];
        init_min_tracked = (int)fs["Initializer.min_tracked"];
        init_min_disparity = (int)fs["Initializer.min_disparity"];
        init_min_inliers = (int)fs["Initializer.min_inliers"];
        init_sigma = (double)fs["Initializer.sigma"];
        init_sigma2 = init_sigma*init_sigma;
        double focal_length2 = fx > fy ? fy*fy : fx*fx;
        init_unsigma2 = init_sigma2 / focal_length2;
        init_unsigma = sqrt(init_unsigma2);
        init_max_iters = (int)fs["Initializer.ransac_max_iters"];

        //! map
        mapping_scale = (double)fs["Mapping.scale"];
        mapping_min_connection_observations = (int)fs["Mapping.min_connection_observations"];
        mapping_min_corners = (int)fs["Mapping.min_corners"];

        //! Align
        align_top_level = (int)fs["Align.top_level"];
        align_top_level = MIN(align_top_level, top_level);
        align_patch_size = (int)fs["Align.patch_size"];

        //! Tracking
        min_quality_fts = (int)fs["Tracking.min_quality_fts"];
        max_quality_drop_fts = (int)fs["Tracking.max_quality_drop_fts"];

        //! glog
        if(!fs["Glog.alsologtostderr"].empty())
            fs["Glog.alsologtostderr"] >> FLAGS_alsologtostderr;

        if(!fs["Glog.colorlogtostderr"].empty())
            fs["Glog.colorlogtostderr"] >> FLAGS_colorlogtostderr;

        if(!fs["Glog.stderrthreshold"].empty())
            fs["Glog.stderrthreshold"] >> FLAGS_stderrthreshold;

        if(!fs["Glog.minloglevel"].empty())
            fs["Glog.minloglevel"] >> FLAGS_minloglevel;

        if(!fs["Glog.log_prefix"].empty())
            fs["Glog.log_prefix"] >> FLAGS_log_prefix;

        if(!fs["Glog.log_dir"].empty())
            fs["Glog.log_dir"] >> FLAGS_log_dir;

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
    int top_level;
    double fps;
    double unit_plane_pixel_length;

    //! FAST detector parameters
//    int image_border;
    int grid_size;
    int grid_min_size;
    //int grid_max_fts;
    int fast_max_threshold;
    int fast_min_threshold;

    double fast_min_eigen;
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

    //! map
    double mapping_scale;
    int mapping_min_connection_observations;
    int mapping_min_corners;

    //! Align
    int align_top_level;
    int align_patch_size;

    //! Tracking
    int min_quality_fts;
    int max_quality_drop_fts;

};

}

#endif