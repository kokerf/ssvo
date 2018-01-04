#include <iostream>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace std;

bool triangulate(const Matrix3d& R_cr,  const Vector3d& t_cr,
                 const Vector3d& fn_r, const Vector3d& fn_c, double &d_ref, double &d_cur)
{
    //! d_c * fn_c = d_r * (R_cr * fn_r) + t_cr
    //! [R_cr*fn_r -fn_c] [d_r d_c]^T = -t_cr, make R_cr*fn_r := Rfn_r
    //! =>
    //! [Rfn_r^T * Rfn_r  -Rfn_r^T * fn_c] [d_r] = [Rfn_r^T * t_cr]
    //! [ fn_c^T * Rfn_r   -fn_c^T * fn_c] [d_c] = [ fn_c^T * t_cr]

    Vector3d R_fn_r(R_cr * fn_r);
    Vector2d b(t_cr.dot(R_fn_r), t_cr.dot(fn_c));
    double A[4] = { R_fn_r.dot(R_fn_r), 0,
                    R_fn_r.dot(fn_c), -fn_c.dot(fn_c)};
    A[1] = -A[2];
    double det = A[0]*A[3] - A[1]*A[2];
   if(std::abs(det) < 0.000001)
       return false;

    d_ref = std::abs((b[0]*A[3] - A[1]*b[1])/det);
    d_cur = std::abs((A[0]*b[1] - b[0]*A[2])/det);

    return true;
}


Vector3d
triangulateFeatureNonLin(const Matrix3d& R,  const Vector3d& t,
                         const Vector3d& feature1, const Vector3d& feature2 )
{
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // => [ f_ref^T f_ref, -f_ref^T f_cur ] [d_ref] = [f_ref^T t]
    //    [ f_cur^T f_ref, -f_cur^T f_cur ] [d_cur] = [f_cur^T t]
    Vector3d f2 = R * feature2;
    Vector2d b;
    b[0] = t.dot(feature1);
    b[1] = t.dot(f2);
    Matrix2d A;
    A(0,0) = feature1.dot(feature1);
    A(1,0) = feature1.dot(f2);
    A(0,1) = -A(1,0);
    A(1,1) = -f2.dot(f2);
    Vector2d lambda = A.inverse() * b;
    Vector3d xm = lambda[0] * feature1;
    Vector3d xn = t + lambda[1] * f2;
    return ( xm + xn )/2;
}

bool
depthFromTriangulationExact(
    const Matrix3d& R_r_c,
    const Vector3d& t_r_c,
    const Vector3d& f_r,
    const Vector3d& f_c,
    double& depth_in_r,
    double& depth_in_c)
{
    // bearing vectors (f_r, f_c) do not need to be unit length
    const Vector3d f_c_in_r(R_r_c*f_c);
    const double a = f_c_in_r.dot(f_r) / t_r_c.dot(f_r);
    const double b = f_c_in_r.dot(t_r_c);
    const double denom = (a*b - f_c_in_r.dot(f_c_in_r));

   if(abs(denom) < 0.000001)
       return false;

    depth_in_c = (b-a*t_r_c.dot(t_r_c)) / denom;
    depth_in_r = (t_r_c + f_c_in_r*depth_in_c).norm();
    return true;
}

bool depthFromTriangulation(
    const Matrix3d& R_c_r,
    const Vector3d& t_c_r,
    const Vector3d& f_ref,
    const Vector3d& f_cur,
    double& depth)
{
    Matrix<double,3,2> A; A << R_c_r * f_ref, f_cur;
    const Matrix2d AtA = A.transpose()*A;
   if(AtA.determinant() < 0.000001)
       return false;
    // d = - (ATA)^(-1) * AT * t
    const Vector2d depth2 = - AtA.inverse()*A.transpose()*t_c_r;
    depth = fabs(depth2[0]);
    return true;
}

int main()
{
    Matrix3d K; K.setIdentity();
    K(0,0) = 414.5;
    K(0,2) = 348.8;
    K(1,1) = 414.2;
    K(1,2) = 240.0;
    Matrix3d Kinv = K.inverse();

    AngleAxisd angle(20.0, Vector3d(0,0,1));
    Matrix3d R_cfr; R_cfr = angle.toRotationMatrix();
    Vector3d t_cfr(0.,0.01,0.01);

    Matrix3d R_rfc = R_cfr.transpose();
    Vector3d t_rfc = -R_rfc*t_cfr;

    std::cout << "R_c_r:\n" << R_cfr << "\n t_c_r:" << t_cfr.transpose() << std::endl;
    std::cout << "R_r_c:\n" << R_rfc << "\n t_r_c:" << t_rfc.transpose() << std::endl;

    const double depth_ref = 1.0;
    const Vector3d px_ref(200, 300, 1);
    const Vector3d fn_ref = Kinv * px_ref;

    Vector3d fn_cur = R_cfr*(depth_ref * fn_ref) + t_cfr;
    const double depth_cur = fn_cur[2];
    Vector3d px_cur = K*fn_cur;
    px_cur /= px_cur[2];
    cout << "Depth ref: " << depth_ref << " cur: " << depth_cur << endl;
    cout << "[ Real] px_ref:" << px_ref.transpose() << " px_cur: " << px_cur.transpose() << endl;

    //! add noise
    px_cur += Vector3d(-1,-1,0);
    fn_cur = Kinv * px_cur;
    cout << "[Noise] px_ref:" << px_ref.transpose() << " px_cur: " << px_cur.transpose() << endl;

    //! Case 1
    Vector3d point_ref = triangulateFeatureNonLin(R_rfc, t_rfc, fn_ref, fn_cur);
    Vector3d point_cur = triangulateFeatureNonLin(R_cfr, t_cfr, fn_cur, fn_ref);
    cout << "[1]  ref: " << point_ref[2] << " cur: " << point_cur[2] << endl;

    //! Case 2
    double d_cur = 0;
    double d_ref = 0;
    depthFromTriangulationExact(R_rfc, t_rfc, fn_ref, fn_cur, d_ref, d_cur);
    cout << "[2]  ref: " << d_ref << " cur: " << d_cur << endl;

    //! Case 3
    double d_ref1 = 0;
    double d_cur1 = 0;
    depthFromTriangulation(R_cfr, t_cfr, fn_ref, fn_cur, d_ref1);
    depthFromTriangulation(R_rfc, t_rfc, fn_cur, fn_ref, d_cur1);
    cout << "[3]  ref: " << d_ref1 << " cur: " << d_cur1 << endl;

    //! Case 4
    double d_ref2 = 0;
    double d_cur2 = 0;
    triangulate(R_cfr, t_cfr, fn_ref, fn_cur, d_ref2, d_cur2);
    cout << "[4]  ref: " << d_ref2 << " cur: " << d_cur2 << endl;

    //====================
    const size_t N = 1000000;
    double time0 = (double)cv::getTickCount();
    for(int i = 0; i < N; ++i)
    {
        triangulateFeatureNonLin(R_rfc, t_rfc, fn_ref, fn_cur);
    }
    double time1 = (double)cv::getTickCount();

    for(int i = 0; i < N; ++i)
    {
        depthFromTriangulationExact(R_rfc, t_rfc, fn_ref, fn_cur, d_ref, d_cur);
    }
    double time2 = (double)cv::getTickCount();


    for(int i = 0; i < N; ++i)
    {
        depthFromTriangulation(R_cfr, t_cfr, fn_ref, fn_cur, d_ref1);
    }
    double time3 = (double)cv::getTickCount();

    for(int i = 0; i < N; ++i)
    {
        triangulate(R_cfr, t_cfr, fn_ref, fn_cur, d_ref2, d_cur2);
    }
    double time4 = (double)cv::getTickCount();

    std::cout << "Time: " << (time1-time0)/cv::getTickFrequency() << " "
              << (time2-time1)/cv::getTickFrequency() << " "
              << (time3-time2)/cv::getTickFrequency() << " "
              << (time4-time3)/cv::getTickFrequency() << " "
              << std::endl;


    return 0;

}