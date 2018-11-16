#include "sim3_solver.hpp"

namespace ssvo{

void Sim3Solver::getEstimateSim3(Matrix3d &R, Vector3d &t, double &s, std::vector<bool> &inliers)
{
    R = R12_best_;
    t = t12_best_;
    s = s12_best_;

    inliers = std::vector<bool>(N_, false);

    for(size_t i = 0; i < indices_.size(); i++)
    {
        if(!inliers_[i])
            continue;

        inliers[indices_[i]] = true;
    }
}

bool Sim3Solver::runRANSAC(const int min_inliers, const int max_iterations)
{
    if(min_inliers > indices_.size())
        return false;

    inliers_ = std::vector<bool>(indices_.size(), false);
    inliers_count_best_ = 0;

    std::vector<int> all_indice(indices_.size(), 0);
    std::iota(all_indice.begin(), all_indice.end(), 0);

    int niters = max_iterations;
    for(int iter = 0; iter < niters; iter++)
    {
        std::vector<Vector3d> mpts1; mpts1.reserve(3);
        std::vector<Vector3d> mpts2; mpts2.reserve(3);
        std::vector<int> samples = all_indice;
        for(int i = 0; i < 3; i++)
        {
            int randi = Rand(0, samples.size()-1);
            mpts1.push_back(mpts1_[samples[randi]]);
            mpts2.push_back(mpts2_[samples[randi]]);

            samples[i] = samples.back();
            samples.pop_back();
        }

        Matrix3d R12;
        Vector3d t12;
        double s12;
        computeSim3(mpts1, mpts2, R12, t12, s12, scale_fixed_);

        std::vector<bool> inliers;
        int inliers_count = checkInliers(R12, t12, s12, inliers);

        if(inliers_count > inliers_count_best_)
        {
            R12_best_ = R12;
            t12_best_ = t12;
            s12_best_ = s12;

            inliers_ = inliers;
            inliers_count_best_ = inliers_count;

            if(inliers_count < min_inliers)
            {
                //! N = log(1-p)/log(1-omega^s)
                //! p = 99%
                //! number of set: s = 8
                //! omega = inlier points / total points
                const static double num = log(1 - 0.99);
                const double omega = inliers_count*1.0 / indices_.size();
                const double denom = log(1 - pow(omega, 3));

                niters = (denom >= 0 || -num >= max_iterations*(-denom)) ? max_iterations : round(num / denom);
            }
            else
                break;
        }

    }

    if(inliers_count_best_ > min_inliers)
        return true;
    else
        return false;
}

//! p1 = s12*R12*p2+t12
//! Sim(3)_12 = [s12R12,t12]
void Sim3Solver::computeSim3(const std::vector<Vector3d> &pts1,
                             const std::vector<Vector3d> &pts2,
                             Matrix3d &R12,
                             Vector3d &t12,
                             double &s12,
                             bool scale_fixed)
{
    Matrix<double, 3, -1> A1 = Eigen::Map<const Matrix<double, 3, -1>>(pts1.data()->data(), 3, pts1.size());
    Matrix<double, 3, -1> A2 = Eigen::Map<const Matrix<double, 3, -1>>(pts2.data()->data(), 3, pts2.size());

    const Vector3d C1 = A1.rowwise().mean();
    const Vector3d C2 = A2.rowwise().mean();

    A1.colwise() -= C1;
    A2.colwise() -= C2;

#if 1

    //! Local Accuracy and Global Consistency for Efficient Visual SLAM, Hauke Strasdat 2012, P129
    Matrix3d H = A2 * A1.transpose();
    JacobiSVD<MatrixXd> svd(H, ComputeFullV|ComputeFullU);
    MatrixXd V = svd.matrixV();
    MatrixXd U = svd.matrixU();
    R12 = V * U.transpose();

#else

    //! Closed-form solution of absolute orientataion using unit quaternions, Horn 1987,
    Matrix3d M = A2 * A1.transpose();
    Matrix4d N = Matrix4d::Zero();
    N(0,0) = M(0,0)+M(1,1)+M(2,2);
    N(0,1) = M(1,2)-M(2,1);
    N(0,2) = M(2,0)-M(0,2);
    N(0,3) = M(0,1)-M(1,0);
    N(1,1) = M(0,0)-M(1,1)-M(2,2);
    N(1,2) = M(0,1)+M(1,0);
    N(1,3) = M(2,0)+M(0,2);
    N(2,2) = -M(0,0)+M(1,1)-M(2,2);
    N(2,3) = M(1,2)+M(2,1);
    N(3,3) = -M(0,0)-M(1,1)+M(2,2);

    N(1,0) = N(0,1);
    N(2,0) = N(0,2);
    N(3,0) = N(0,3);
    N(2,1) = N(1,2);
    N(3,1) = N(1,3);
    N(3,2) = N(2,3);

    SelfAdjointEigenSolver<Matrix4d> eig(N);
    Vector4d eig_values = eig.eigenvalues();
    Matrix4d eig_vectors = eig.eigenvectors();

    int max_idx = -1;
    eig_values.maxCoeff(&max_idx);

    Vector4d evec = eig_vectors.col(max_idx);

    //double angle = std::atan2(evec.tail<3>().norm(), evec[0]);
    //R12 = AngleAxisd(2*angle, evec.tail<3>()/evec.tail<3>().norm()).toRotationMatrix();

    R12 = Quaterniond(evec[0], evec[1], evec[2], evec[3]).toRotationMatrix();

#endif

    if(!scale_fixed)
        s12 = std::sqrt(A1.rowwise().squaredNorm().sum() / A2.rowwise().squaredNorm().sum());
    else
        s12 = 1.0;

    t12 = C1 - s12*R12 * C2;
}

int Sim3Solver::checkInliers(Matrix3d R12, Vector3d t12, double s12, std::vector<bool>& inliers)
{
    const Matrix3d sR12 = s12 * R12;
    const Matrix3d sR21 = (1.0/s12) * R12.transpose();
    const Vector3d t21 =  sR21 * (-t12);

    int inlier_count = 0;
    inliers.resize(indices_.size(), false);
    for(size_t i = 0; i < indices_.size(); i++)
    {
        const Vector3d px12 = sR12 * mpts2_[i] + t12;
        const Vector3d px21 = sR21 * mpts1_[i] + t21;

        const Vector2d dist12 = px12.head<2>() / px12[2] - pxls1_[i];
        const Vector2d dist21 = px21.head<2>() / px21[2] - pxls2_[i];

        const double err12 = dist12.squaredNorm();
        const double err21 = dist21.squaredNorm();

        if(err12 < max_err1_[i] && err21 < max_err2_[i])
        {
            inliers[i] = true;
            inlier_count++;
        }
        else
            inliers[i] = false;

    }

    return inlier_count;
}

}

