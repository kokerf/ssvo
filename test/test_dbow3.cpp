#include <opencv2/opencv.hpp>
#include <DBoW3/Vocabulary.h>
#include <DBoW3/Database.h>
#include <DBoW3/DescManip.h>
#include "brief.hpp"

std::vector<std::string> readImagePaths(int argc,char **argv,int start){
    std::vector<std::string> paths;
    for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
    return paths;
}

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

int main(int argc, char *argv[])
{

    if(argc<=3){
        std::cerr<<"Usage: ./test_dbow3 voc_file image1 image2 ..."<< std::endl;
        return -1;
    }

    std::vector<std::string> img_path = readImagePaths(argc, argv, 2);

    DBoW3::Vocabulary voc(argv[1]);
    DBoW3::Database db(voc, true, 4);
    std::cout << "=========" << std::endl;
    std::cout << "Voc and DB info:" << std::endl;
    std::cout << voc << std::endl;
    std::cout << db << std::endl;


    std::vector<cv::Mat> images;
    for(int i = 0; i < img_path.size(); ++i)
    {
        cv::Mat img = cv::imread(img_path[i], CV_LOAD_IMAGE_GRAYSCALE);
        if(img.empty())
        {
            std::cerr << " Cant load image: " << img_path[i] << std::endl;
            return false;
        }
        images.push_back(img);
    }

    std::vector<std::vector<cv::KeyPoint> > kps(images.size());
    std::vector<cv::Mat > desps(images.size());
    std::vector<cv::Mat > desps1(images.size());

    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 2, 1);
    ssvo::BRIEF brief;
    double t0 = cv::getTickCount();
    for(int i = 0; i < images.size(); ++i)
    {
        orb->detect(images[i], kps[i]);
    }

    double t1 = cv::getTickCount();
    for(int i = 0; i < images.size(); ++i)
    {
        orb->compute(images[i], kps[i], desps[i]);
    }

    double t2 = cv::getTickCount();
    for(int i = 0; i < images.size(); ++i)
    {
        std::vector<cv::Mat> imgPyr;
        cv::buildPyramid(images[0], imgPyr, 0);
        brief.compute(imgPyr, kps[i], desps1[i]);
    }

    double t3 = cv::getTickCount();
    std::cout << "Time: " << (t1-t0)/cv::getTickFrequency()/images.size()
        << ", " << (t2-t1)/cv::getTickFrequency()/images.size()
        << ", " << (t3-t2)/cv::getTickFrequency()/images.size()  << std::endl;


    for(int j = 0; j < kps.size(); ++j)
    {
        std::cout << "1: " << to_binary(desps[0].row(j)) << std::endl;
        std::cout << "2: " << to_binary(desps1[0].row(j)) << std::endl;
    }

    std::vector<DBoW3::BowVector> bvs(images.size());
    std::vector<DBoW3::FeatureVector> fvs(images.size());
    double t4 = cv::getTickCount();
    for(int i = 0; i < images.size(); ++i)
    {
        db.add(desps[i], &bvs[i], &fvs[i]);
    }
    double t5 = cv::getTickCount();
    std::cout << "Database Time: " << (t5-t4)/cv::getTickFrequency()/images.size() << std::endl;

    std::cout << "\n=========" << std::endl;
    std::cout << "* BowVector of image 0:\n" << bvs[0] << std::endl;
    std::cout << "* FeatureVector of image 0:\n" << fvs[0] << std::endl;

    std::vector<uint> ft = fvs[0].begin()->second;
    uint node_id = fvs[0].begin()->first;
    std::cout << "\n=========" << std::endl;
    std::cout << "* desp of node: " << node_id << "\n[" << to_binary(voc.getNode(node_id)) << "]" << std::endl;
    for(int i = 0; i < ft.size(); ++i)
    {

        std::cout << "desp of kp:" << ft[i] << " in image0, "
            << "dis: " << DBoW3::DescManip::distance(voc.getNode(node_id), desps[0].row(ft[i]))
            << "\n[" << to_binary(desps[0].row(ft[i])) << "]" << std::endl;
    }

    // and query the database
    std::cout << "\n=========" << std::endl;
    std::cout << "* Querying the database: " << std::endl;

    DBoW3::QueryResults ret;
    for(size_t i = 0; i < bvs.size(); i++)
    {
        db.query(bvs[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        std::cout << "Searching for Image " << i << ". " << ret << std::endl;
    }

    for(int j = 0; j < images.size(); ++j)
    {
        std::string id = std::to_string(j);
        cv::Mat show;
        cv::drawKeypoints(images[j], kps[j], show);
        cv::imshow("img"+id, show);
    }
    cv::waitKey(0);

    std::cout << std::endl;

    return 0;
}
