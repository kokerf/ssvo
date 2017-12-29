#include "feature.hpp"
#include "utils.hpp"

namespace ssvo{

//! Feature
Feature::Feature(const Vector2d &px, const Vector3d &fn, const int level, const std::shared_ptr<MapPoint> &mpt):
    px(px), fn(fn), level(level), type(STABLE), mpt(mpt)
{
    if(mpt == nullptr) type = CANDIDATE;
}

//! Seed
Seed::Seed(const Feature::Ptr &ft, double depth_mean, double depth_min) :
    ft(ft),
    a(10),
    b(10),
    mu(1.0/depth_mean),
    z_range(1.0/depth_min),
    sigma2(z_range*z_range/36)
{}

double Seed::computeTau(
    const SE3d& T_ref_cur,
    const Vector3d& f,
    const double z,
    const double px_error_angle)
{
    Vector3d t(T_ref_cur.translation());
    Vector3d a = f*z-t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(-a.dot(t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + px_error_angle;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines

    double tau = z_plus - z;
    return 0.5 * (1.0/MAX(0.0000001, z-tau) - 1.0/(z+tau));
}

void Seed::update(const double x, const double tau2)
{
    double norm_scale = sqrt(sigma2 + tau2);
    if(std::isnan(norm_scale))
        return;

    double s2 = 1./(1./sigma2 + 1./tau2);
    double m = s2*(mu/sigma2 + x/tau2);
    double C1 = a/(a+b) * utils::normal_distribution<double>(x, mu, norm_scale);
    double C2 = b/(a+b) * 1./z_range;
    double normalization_constant = C1 + C2;
    C1 /= normalization_constant;
    C2 /= normalization_constant;
    double f = C1*(a+1.)/(a+b+1.) + C2*a/(a+b+1.);
    double e = C1*(a+1.)*(a+2.)/((a+b+1.)*(a+b+2.))
        + C2*a*(a+1.0f)/((a+b+1.0f)*(a+b+2.0f));

    // update parameters
    double mu_new = C1*m+C2*mu;
    sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new;
    mu = mu_new;
    a = (e-f)/(f-e/f);
    b = a*(1.0f-f)/f;
}

}
