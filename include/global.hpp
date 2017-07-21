#ifndef _GLOBAL_HPP_
#define _GLOBAL_HPP_

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <stdint.h>

inline double Rand(double min, double max)
{ return (((double)rand()/((double)RAND_MAX + 1.0)) * (max - min + 1)) + min;}

inline int Rand(int min, int max)
{ return (((double)rand()/((double)RAND_MAX + 1.0)) * (max - min + 1)) + min;}

#ifndef MIN
    #define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
    #define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

#ifdef SSVO_IMFORM_OUTPUT
    #define SSVO_WARN_STREAM(x) std::cerr<<"\033[0;33m[WARN] "<<x<<"\033[0;0m"<<std::endl;
    #define SSVO_ERROR_STREAM(x) std::cerr<<"\033[1;31m[ERROR] "<<x<<"\033[0;0m"<<std::endl;
    #define SSVO_INFO_STREAM(x) std::cerr<<"\033[0;0m[INFO] "<<x<<"\033[0;0m"<<std::endl;
#else
    #define SSVO_WARN_STREAM(x)
    #define SSVO_ERROR_STREAM(x)
    #define SSVO_INFO_STREAM(x)
#endif

typedef std::vector<cv::Mat> ImgPyr;

#endif