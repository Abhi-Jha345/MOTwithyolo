#include "tracker.hpp"
#include <algorithm>
#include <numeric>
#include <limits>

// KalmanBox

void KalmanBox::init(float x1, float y1, float x2, float y2) {
    kf.init(8, 4, 0, CV_32F);

    // Transition matrix (constant velocity)
    kf.transitionMatrix = (cv::Mat_<float>(8,8) <<
        1,0,0,0, 1,0,0,0,
        0,1,0,0, 0,1,0,0,
        0,0,1,0, 0,0,1,0,
        0,0,0,1, 0,0,0,1,
        0,0,0,0, 1,0,0,0,
        0,0,0,0, 0,1,0,0,
        0,0,0,0, 0,0,1,0,
        0,0,0,0, 0,0,0,1);

    // Measurement matrix (observe cx,cy,w,h)
    kf.measurementMatrix = cv::Mat::zeros(4, 8, CV_32F);
    kf.measurementMatrix.at<float>(0,0) = 1;
    kf.measurementMatrix.at<float>(1,1) = 1;
    kf.measurementMatrix.at<float>(2,2) = 1;
    kf.measurementMatrix.at<float>(3,3) = 1;

    // Low process noise = smooth motion; high measurement noise = damp detection jitter.
    // Size (w,h) gets extra-high measurement noise since YOLO box size pulses most.
    kf.processNoiseCov     = cv::Mat::eye(8, 8, CV_32F) * 1e-3f;
    kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 10.0f;
    kf.measurementNoiseCov.at<float>(2,2) = 50.0f;   // width  - heavily damp
    kf.measurementNoiseCov.at<float>(3,3) = 50.0f;   // height - heavily damp
    kf.errorCovPost        = cv::Mat::eye(8, 8, CV_32F) * 10.f;

    float cx = (x1+x2)/2, cy = (y1+y2)/2, w = x2-x1, h = y2-y1;
    kf.statePost = (cv::Mat_<float>(8,1) << cx,cy,w,h, 0,0,0,0);
    initialized = true;
}

cv::Rect2f KalmanBox::predict() {
    auto s = kf.predict();
    return state();
}

cv::Rect2f KalmanBox::update(float x1, float y1, float x2, float y2) {
    float cx=(x1+x2)/2, cy=(y1+y2)/2, w=x2-x1, h=y2-y1;
    cv::Mat meas = (cv::Mat_<float>(4,1) << cx,cy,w,h);
    kf.correct(meas);
    return state();
}

cv::Rect2f KalmanBox::state() const {
    const auto& s = kf.statePost;
    float cx=s.at<float>(0), cy=s.at<float>(1);
    float w=s.at<float>(2),  h=s.at<float>(3);
    return {cx-w/2, cy-h/2, w, h};
}

// Tracker

Tracker::Tracker(int min_hits, int max_age, float iou_thresh)
    : min_hits_(min_hits), max_age_(max_age), iou_thresh_(iou_thresh) {}

float Tracker::iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Rect2f inter = a & b;
    float inter_area = inter.area();
    if (inter_area <= 0) return 0.f;
    return inter_area / (a.area() + b.area() - inter_area);
}

std::vector<std::pair<int,int>> Tracker::hungarian(
    const std::vector<std::vector<float>>& cost,
    const std::vector<int>& row_ids,
    const std::vector<int>& col_ids,
    float thresh)
{
    // Greedy O(n^2) - adequate for small n
    int nr = row_ids.size(), nc = col_ids.size();
    std::vector<bool> used_col(nc, false);
    std::vector<std::pair<int,int>> matches;

    for (int r = 0; r < nr; r++) {
        float best = thresh;
        int best_c = -1;
        for (int c = 0; c < nc; c++) {
            if (!used_col[c] && cost[r][c] < best) {
                best = cost[r][c];
                best_c = c;
            }
        }
        if (best_c >= 0) {
            matches.push_back({row_ids[r], col_ids[best_c]});
            used_col[best_c] = true;
        }
    }
    return matches;
}

void Tracker::update(const std::vector<Detection>& dets) {
    // 1. Predict all tracks
    for (auto& [id, trk] : tracks_)
        trk.bbox = trk.kalman.predict();

    // 2. Build cost matrix (1 - IoU)
    std::vector<int> trk_ids, det_ids;
    for (auto& [id, _] : tracks_) trk_ids.push_back(id);
    for (int i = 0; i < (int)dets.size(); i++) det_ids.push_back(i);

    std::vector<std::vector<float>> cost(trk_ids.size(), std::vector<float>(dets.size(), 1.f));
    for (int r = 0; r < (int)trk_ids.size(); r++) {
        const auto& tb = tracks_.at(trk_ids[r]).bbox;
        for (int c = 0; c < (int)dets.size(); c++) {
            cv::Rect2f db(dets[c].x1, dets[c].y1,
                          dets[c].x2-dets[c].x1, dets[c].y2-dets[c].y1);
            cost[r][c] = 1.f - iou(tb, db);
        }
    }

    // 3. Match
    auto matches = hungarian(cost, trk_ids, det_ids, 1.f - iou_thresh_);

    std::set<int> matched_trks, matched_dets;
    // Separate smoothing: position moves, but size should be near-constant
    constexpr float EMA_POS  = 0.7f;   // box position smoothing
    constexpr float EMA_SIZE = 0.9f;   // box size smoothing (heavier - size pulses most)
    for (auto [tid, did] : matches) {
        matched_trks.insert(tid);
        matched_dets.insert(did);
        auto& d = dets[did];
        auto& trk = tracks_.at(tid);
        trk.bbox = trk.kalman.update(d.x1, d.y1, d.x2, d.y2);
        if (!trk.smooth_init) { trk.smooth_bbox = trk.bbox; trk.smooth_init = true; }
        else {
            trk.smooth_bbox.x      = EMA_POS *trk.smooth_bbox.x      + (1-EMA_POS )*trk.bbox.x;
            trk.smooth_bbox.y      = EMA_POS *trk.smooth_bbox.y      + (1-EMA_POS )*trk.bbox.y;
            trk.smooth_bbox.width  = EMA_SIZE*trk.smooth_bbox.width  + (1-EMA_SIZE)*trk.bbox.width;
            trk.smooth_bbox.height = EMA_SIZE*trk.smooth_bbox.height + (1-EMA_SIZE)*trk.bbox.height;
        }
        trk.hits++;
        trk.misses = 0;
        if (trk.hits >= min_hits_) trk.confirmed = true;
    }

    // 4. Unmatched tracks - age them
    for (auto& [id, trk] : tracks_) {
        if (!matched_trks.count(id)) trk.misses++;
    }

    // 5. New tracks from unmatched detections
    for (int i = 0; i < (int)dets.size(); i++) {
        if (matched_dets.count(i)) continue;
        const auto& d = dets[i];
        Track t;
        t.id = next_id_++;
        t.kalman.init(d.x1, d.y1, d.x2, d.y2);
        t.bbox  = {d.x1, d.y1, d.x2-d.x1, d.y2-d.y1};
        t.hits  = 1;
        tracks_[t.id] = t;
    }

    // 6. Remove dead tracks
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        if (it->second.misses > max_age_) it = tracks_.erase(it);
        else ++it;
    }
}

void Tracker::propagate(const cv::Mat& prev_gray, const cv::Mat& curr_gray) {
    for (auto& [id, trk] : tracks_) {
        if (!trk.confirmed) continue;

        // Detect new keypoints if too few
        cv::Rect2f b = trk.bbox;
        cv::Rect roi(std::max(0.f,b.x), std::max(0.f,b.y),
                     std::min((float)prev_gray.cols-b.x, b.width),
                     std::min((float)prev_gray.rows-b.y, b.height));
        if (roi.area() < 1) continue;

        if (trk.keypoints.size() < 10) {
            cv::goodFeaturesToTrack(prev_gray(roi), trk.keypoints, 50, 0.05, 3);
            for (auto& p : trk.keypoints) { p.x += roi.x; p.y += roi.y; }
        }
        if (trk.keypoints.empty()) continue;

        // Track points
        std::vector<cv::Point2f> next_pts;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray, curr_gray, trk.keypoints, next_pts,
                                 status, err, cv::Size(5,5), 3);

        // Keep good points, compute translation
        std::vector<cv::Point2f> good_prev, good_next;
        for (int i = 0; i < (int)status.size(); i++) {
            if (status[i] && err[i] < 20.f) {
                good_prev.push_back(trk.keypoints[i]);
                good_next.push_back(next_pts[i]);
            }
        }
        trk.keypoints = good_next;

        if (good_prev.size() < 3) continue;

        // Estimate rigid motion (translation + scale)
        cv::Mat aff = cv::estimateAffinePartial2D(good_prev, good_next);
        if (aff.empty()) continue;

        float dx  = aff.at<double>(0,2);
        float dy  = aff.at<double>(1,2);
        float sc  = std::sqrt(aff.at<double>(0,0)*aff.at<double>(0,0) +
                               aff.at<double>(1,0)*aff.at<double>(1,0));
        sc = std::clamp(sc, 0.9f, 1.1f);

        trk.bbox.x += dx;
        trk.bbox.y += dy;
        trk.bbox.width  *= sc;
        trk.bbox.height *= sc;
        // Propagate smooth bbox too
        constexpr float KLT_EMA = 0.7f;
        if (trk.smooth_init) {
            trk.smooth_bbox.x      = KLT_EMA*trk.smooth_bbox.x      + (1-KLT_EMA)*trk.bbox.x;
            trk.smooth_bbox.y      = KLT_EMA*trk.smooth_bbox.y      + (1-KLT_EMA)*trk.bbox.y;
            trk.smooth_bbox.width  = KLT_EMA*trk.smooth_bbox.width  + (1-KLT_EMA)*trk.bbox.width;
            trk.smooth_bbox.height = KLT_EMA*trk.smooth_bbox.height + (1-KLT_EMA)*trk.bbox.height;
        }
    }
}
