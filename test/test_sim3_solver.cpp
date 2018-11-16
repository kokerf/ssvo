#include "sim3_solver.hpp"

int main()
{
    std::vector<Vector3d> pts1;
    std::vector<Vector3d> pts2;

    pts2.push_back(Vector3d(1,1,1));
    pts2.push_back(Vector3d(2,10,4));
    pts2.push_back(Vector3d(5,5,8));
    pts2.push_back(Vector3d(3,3,5));
    pts2.push_back(Vector3d(0,32,5));
    pts2.push_back(Vector3d(51,64,3));
    pts2.push_back(Vector3d(10,9,83));


    Quaterniond q12(1,0,0.9,0);
    Matrix3d R12 = q12.toRotationMatrix();
    Vector3d t12 = Vector3d(3,5,2);
    double s12 = 10;

    std::cout << "R12:\n " << R12 << std::endl;
    std::cout << "t12:\n " << t12 << "\ns12: " << s12 << std::endl;

    pts1.resize(pts2.size());
    for(size_t i = 0; i < pts1.size(); i++)
    {
        pts1[i] = s12*R12*pts2[i]+t12;
    }

    Matrix3d R12_;
    Vector3d t12_;
    double s12_;
    ssvo::Sim3Solver::computeSim3(pts1, pts2, R12_, t12_, s12_);

    std::cout << "R12:\n " << R12_  << ", " << R12_.determinant()<< std::endl;
    std::cout << "t12:\n " << t12_ << "\ns12: " << s12_ << std::endl;

    std::vector<double> error(pts1.size(),0);
    for(size_t i = 0; i < pts1.size(); i++)
    {
        error[i] = (pts1[i] - s12_*R12_*pts2[i]+t12_).norm();
    }

    return 0;
}
