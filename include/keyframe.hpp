#ifndef _KEYFRAME_HPP_
#define _KEYFRAME_HPP_

#include <DBoW3/DBoW3.h>
#include <DBoW3/DescManip.h>
#include "global.hpp"
#include "frame.hpp"
#include "feature_detector.hpp"
#include "brief.hpp"

namespace ssvo
{

class Map;

class KeyFrame: public Frame, public std::enable_shared_from_this<KeyFrame>
{
public:

    typedef std::shared_ptr<KeyFrame> Ptr;

    enum {
        GRID_ROWS = 48,
        GRID_COLS = 64,
    };

    //! about features
    size_t N();

    bool addMapPoint(const MapPoint::Ptr& mpt, const size_t &idx);

    const Feature::Ptr& getFeatureByIndex(const size_t &idx);

    const MapPoint::Ptr& getMapPointByIndex(const size_t &idx);

    bool addSeedFeatureCreated(const Seed::Ptr &seed, const size_t &idx);

    bool removeSeedCreateByIndex(const size_t &idx);

    std::vector<size_t> getSeedCreateIndices();

//    bool removeMapPointMatchByMapPoint(const MapPoint::Ptr &mpt);

    void detectFast(const FastDetector::Ptr &fast);

    void conputeDescriptor(const BRIEF::Ptr &brief);

    void computeBoW(const DBoW3::Vocabulary& vocabulary);

    std::vector<size_t> getFeaturesInArea(const float x, const float y, const float r, const int min_level = -1, const int max_level = -1);

    std::vector<size_t> getFeaturesInGrid(const int r, const int c) const;

    //! keyframes
    void setBad();

    bool isBad();

    void updateConnections();

    std::set<KeyFrame::Ptr> getConnectedKeyFrames(int num=-1, int min_fts = 0);

    std::set<KeyFrame::Ptr> getSubConnectedKeyFrames(int num=-1);

    //! deleted funcs
    const ImgPyr opticalImages() = delete;    //! disable this function

    bool addMapPointFeatureMatch(const MapPoint::Ptr &mpt, const Feature::Ptr &ft) = delete;

    bool addSeedFeatureMatch(const Seed::Ptr &seed, const Feature::Ptr &ft) = delete;

    //! static funcs
    inline static KeyFrame::Ptr create(const Frame::Ptr frame)
    { return Ptr(new KeyFrame(frame)); }

private:

    KeyFrame(const Frame::Ptr frame);

    void addConnection(const KeyFrame::Ptr &kf, const int weight);

    void updateOrderedConnections();

    void removeConnection(const KeyFrame::Ptr &kf);

    void assignFeaturesToGrid();

    inline bool getGridPos(const float x, const float y, int &pos_x, int &pos_y);

public:

    static uint64_t next_id_;

    const uint64_t frame_id_;

    std::vector<cv::Mat> descriptors_;

    DBoW3::BowVector bow_vec_;

    DBoW3::FeatureVector feat_vec_;

    unsigned int dbow_Id_;

private:

    float grid_col_inv_;
    float grid_row_inv_;

    size_t N_;
    std::unordered_set<size_t> seeds_created_;

    std::vector<std::size_t> grid_[GRID_ROWS][GRID_COLS];

    std::map<KeyFrame::Ptr, int> connectedKeyFrames_;

    std::multimap<int, KeyFrame::Ptr> orderedConnectedKeyFrames_;

    bool isBad_;

    std::mutex mutex_connection_;

};

}

#endif