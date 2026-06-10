#pragma once
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "detector.hpp"

// Kalman filter (constant-velocity 2D bbox)
// State:   [cx, cy, w, h, vx, vy, vw, vh]
// Measure: [cx, cy, w, h]
struct KalmanBox {
    cv::KalmanFilter kf;
    bool initialized = false;

    void init(float x1, float y1, float x2, float y2);
    cv::Rect2f predict();
    cv::Rect2f update(float x1, float y1, float x2, float y2);
    cv::Rect2f state() const;
};

// Track
struct Track {
    int   id;
    int   hits       = 0;
    cv::Rect2f smooth_bbox;       // EMA-smoothed bbox for display
    bool  smooth_init = false;
    int   misses     = 0;
    bool  confirmed  = false;
    cv::Rect2f bbox;               // current predicted bbox
    KalmanBox kalman;
    std::vector<cv::Point2f> keypoints;
};

// Simple IoU tracker (SORT-style)
class Tracker {
public:
    explicit Tracker(int min_hits = 3, int max_age = 30, float iou_thresh = 0.3f);

    // Update with new detections; returns confirmed tracks
    void update(const std::vector<Detection>& dets);

    // Propagate tracks using KLT optical flow (between detector frames)
    void propagate(const cv::Mat& prev_gray, const cv::Mat& curr_gray);

    const std::map<int, Track>& tracks() const { return tracks_; }

private:
    static float iou(const cv::Rect2f& a, const cv::Rect2f& b);
    static Detection rectToDet(const cv::Rect2f& r);

    // Hungarian assignment (minimise cost)
    std::vector<std::pair<int,int>> hungarian(
        const std::vector<std::vector<float>>& cost,
        const std::vector<int>& row_ids,
        const std::vector<int>& col_ids,
        float thresh);

    std::map<int, Track> tracks_;
    int next_id_    = 1;
    int min_hits_   = 3;
    int max_age_    = 30;
    float iou_thresh_ = 0.3f;
};
