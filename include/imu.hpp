#ifndef _SSVO_IMU_HPP_
#define _SSVO_IMU_HPP_

#include <deque>
#include <mutex>
#include <memory>
#include <Eigen/Core>

#include "global.hpp"

namespace ssvo{

using namespace Eigen;

struct IMUData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IMUData(double _timestamp, Vector3d _gyro, Vector3d _acc) :
        timestamp(_timestamp), gyro(_gyro), acc(_acc) {}

    double timestamp;
    Vector3d gyro;
    Vector3d acc;
};

class IMUProcessor : public noncopyable
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef std::shared_ptr<IMUProcessor> Ptr;

    bool isInitialized() const;

    const Vector3d gyroBais() const {return gyro_bais_;}

    void addIMUData(const IMUData &data);

    void getIMUData(const double timestamp, std::vector<IMUData> &imu_data);

    bool initializeGravityAndBias();

    inline static Ptr create()
    {return Ptr(new IMUProcessor());}

private:

    IMUProcessor();

private:

    bool is_initialed_;
    Vector3d gyro_bais_;
    Vector3d acc_bais_;
    Vector3d gravity_;

    std::deque<IMUData> imu_buffer_;

    std::mutex mutex_imu_;
};

}

#endif //_SSVO_IMU_HPP_
