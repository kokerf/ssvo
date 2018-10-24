#include <opencv2/opencv.hpp>
#include <DBoW3/Vocabulary.h>
#include <DBoW3/Database.h>
#include <DBoW3/DescManip.h>
#include "brief.hpp"
#include "feature_tracker.hpp"
#include "time_tracing.hpp"

std::string to_binary(const cv::Mat &desp)
{
    assert(desp.cols == 1 || desp.rows == 1);
    assert(desp.type() == CV_8UC1);
    const int size = desp.cols * desp.rows;
    const uchar* pt = desp.ptr<uchar>(0);
    std::string out;
    for(int i = 0; i < size; i++, pt++)
    {
        for(int j = 0; j < 8; ++j)
        {
            out += *pt & (0x01<<j) ? '1' : '0';
        }

        if(i < size-1)
            out += " ";
    }
    return out;
}

void matches_by_dbow(const std::vector<cv::Mat> &desp1, const std::vector<cv::Mat> &desp2,
                     const DBoW3::FeatureVector & feat_vec1, const DBoW3::FeatureVector &feat_vec2,
                    std::vector<cv::DMatch> &matches)
{
    DBoW3::FeatureVector::const_iterator ft1_itr = feat_vec1.begin();
    DBoW3::FeatureVector::const_iterator ft2_itr = feat_vec2.begin();
    DBoW3::FeatureVector::const_iterator ft1_end = feat_vec1.end();
    DBoW3::FeatureVector::const_iterator ft2_end = feat_vec2.end();

    std::vector<size_t> is_matches(desp2.size(), -1);
    std::vector<int> dist_matches(desp2.size(), 255);

    while(ft1_itr != ft1_end && ft2_itr != ft2_end)
    {
        if(ft1_itr->first == ft2_itr->first)
        {
            for(const size_t &idx1 : ft1_itr->second)
            {
                //! ft1
                const cv::Mat &descriptor1 = desp1[idx1];

                int best_dist = 50;
                int best_idx2 = -1;
                for(const size_t &idx2 : ft2_itr->second)
                {
                    //! ft2
                    const cv::Mat &descriptor2 = desp2[idx2];

                    const int dist = DBoW3::DescManip::distance_8uc1(descriptor1, descriptor2);
                    if(dist > best_dist)
                        continue;

                    if(dist < best_dist)
                    {
                        best_dist = dist;
                        best_idx2 = idx2;
                    }
                }


                if(best_idx2 > 0)
                {
//                    const Feature::Ptr ft2 = keyframe2->getFeatureByIndex(best_idx2);
//                    showEplMatch(keyframe1->getImage(0), keyframe2->getImage(0), F12, ft1->px_, ft2->px_);
                    if(is_matches[best_idx2] != -1 && best_dist >= dist_matches[best_idx2])
                    {
                        //std::cout << "error" << std::endl;
                        continue;
                    }

                    matches.emplace_back(cv::DMatch(idx1, best_idx2, best_dist));

                    is_matches[best_idx2] = idx1;
                    dist_matches[best_idx2] = best_dist;
                }
            }

            ft1_itr++;
            ft2_itr++;
        }
        else if(ft1_itr->first < ft2_itr->first)
        {
            ft1_itr = feat_vec1.lower_bound(ft2_itr->first);
        }
        else
        {
            ft2_itr = feat_vec2.lower_bound(ft1_itr->first);
        }
    }
}

int main(int argc, char *argv[])
{

    if(argc<=3){
        std::cerr<<"Usage: ./test_dbow3 voc_file image1 image2"<< std::endl;
        return -1;
    }

    DBoW3::Vocabulary voc(argv[1]);
    DBoW3::Database db(voc, true, 4);
    std::cout << "=========" << std::endl;
    std::cout << "Voc and DB info:" << std::endl;
    std::cout << voc << std::endl;
    std::cout << db << std::endl;

    cv::Mat img1 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);

    ssvo::FastDetector::Ptr fast_detector = ssvo::FastDetector::create(img1.cols, img2.rows, 8, 8, 1.2, 64, 8);

    ssvo::Frame::initScaleParameters(fast_detector);
    ssvo::Frame::Ptr frame1 = ssvo::Frame::create(img1, -1, nullptr);
    ssvo::Frame::Ptr frame2 = ssvo::Frame::create(img2, -1, nullptr);

    std::vector<ssvo::Corner> old_corners;
    std::vector<ssvo::Corner> corners1, corners2;
    fast_detector->detect(frame1->images(), corners1, old_corners, 1000);
    fast_detector->detect(frame2->images(), corners2, old_corners, 1000);

    std::vector<cv::KeyPoint> kps1, kps2;
    kps1.reserve(corners1.size());
    kps2.reserve(corners2.size());
    for(const ssvo::Corner & corner : corners1)
        kps1.emplace_back(cv::KeyPoint(corner.x, corner.y, 31, -1, 0, corner.level));
    for(const ssvo::Corner & corner : corners2)
        kps2.emplace_back(cv::KeyPoint(corner.x, corner.y, 31, -1, 0, corner.level));


    ssvo::BRIEF::Ptr brief = ssvo::BRIEF::create(fast_detector->getScaleFactor(), fast_detector->getNLevels());
    cv::Mat desp1, desp2;
    brief->compute(frame1->images(), kps1, desp1);
    brief->compute(frame2->images(), kps2, desp2);
    std::vector<cv::Mat> desp_vec1, desp_vec2;
    desp_vec1.reserve(desp1.rows);
    desp_vec2.reserve(desp2.rows);
    for(int i = 0; i < desp1.rows; i++)
        desp_vec1.push_back(desp1.row(i));
    for(int i = 0; i < desp2.rows; i++)
        desp_vec2.push_back(desp2.row(i));

    cv::Mat kp_show1, kp_show2;
    cv::drawKeypoints(img1, kps1, kp_show1);
    cv::drawKeypoints(img2, kps2, kp_show2);
    cv::imshow("kp1", kp_show1);
    cv::imshow("kp2", kp_show2);

    cv::waitKey(0);

    const int N = 1000;
    std::vector<cv::DMatch> matches;
    cv::Mat match_show;

    ssvo::SecondTimer timer;
    double duration[3] = {0};

    {
        DBoW3::BowVector bow_vec1, bow_vec2;
        DBoW3::FeatureVector feat_vec1, feat_vec2;
        voc.transform(desp_vec1, bow_vec1, feat_vec1, 4);

        timer.start();
        for(int i = 0; i < N; i++)
        {
            voc.transform(desp_vec2, bow_vec2, feat_vec2, 4);
        }
        duration[0] = timer.stop();

        timer.start();
        for(int i = 0; i < N; i++)
        {
            matches.clear();
            matches.reserve(kps1.size());
            matches_by_dbow(desp_vec1, desp_vec2, feat_vec1, feat_vec2, matches);
        }
        duration[1] = timer.stop();

        cv::drawMatches(img1, kps1, img2, kps2, matches, match_show);

        cv::imshow("dbow_match", match_show);
        cv::waitKey(0);
    }

    cv::BFMatcher bfMatcher = cv::BFMatcher(cv::Hamming::normType, true);
    {
        timer.start();
        for(int i = 0; i < N; i++)
        {
            std::vector<cv::DMatch> matches_temp;
            bfMatcher.match(desp1, desp2, matches_temp);
            matches.clear();
            matches.reserve(matches_temp.size());
            for(const auto m : matches_temp)
            {
                if(m.distance < 50) matches.push_back(m);
            }
        }
        duration[2] = timer.stop();

        cv::drawMatches(img1, kps1, img2, kps2, matches, match_show);
        cv::imshow("bf_match", match_show);
        cv::waitKey(0);
    }

    std::cout << "Dbow time cost: " << duration[0]/N << " + " << duration[1]/N << "\n"
              << "Bf   time cost: " << duration[2]/N << std::endl;

    return 0;
}
