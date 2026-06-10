#include <iostream>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include "detector.hpp"
#include "tracker.hpp"

// Colour palette for track IDs
static cv::Scalar idColor(int id) {
    static const cv::Scalar palette[] = {
        {0,255,0},{0,128,255},{255,0,128},{255,255,0},{0,255,255},{255,0,255},
        {128,255,0},{0,255,128},{255,128,0},{128,0,255}
    };
    return palette[id % 10];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine.path> <input.mp4> [output.mp4] [conf=0.35] [skip=5]\n";
        return 1;
    }

    std::string engine_path = argv[1];
    std::string input_path  = argv[2];
    std::string output_path = (argc > 3) ? argv[3] : "";
    float conf_thresh = (argc > 4) ? std::stof(argv[4]) : 0.35f;
    int   det_skip    = (argc > 5) ? std::stoi(argv[5]) : 5;

    // Init
    std::cout << "Loading TRT engine: " << engine_path << "\n";
    Detector  detector(engine_path, conf_thresh);
    Tracker   tracker(3, 30, 0.3f);

    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "Cannot open: " << input_path << "\n"; return 1; }

    int W   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int H   = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = (int)cap.get(cv::CAP_PROP_FPS);
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "Video: " << W << "x" << H << " @ " << fps << " fps  "
              << total_frames << " frames\n";

    cv::VideoWriter writer;
    if (!output_path.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
        writer.open(output_path, fourcc, fps, {W, H});
    }

    // Main loop
    cv::Mat frame, prev_gray, curr_gray;
    int frame_idx = 0;
    long long total_ns = 0;
    auto t_start = std::chrono::steady_clock::now();

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        auto t0 = std::chrono::steady_clock::now();

        cv::cvtColor(frame, curr_gray, cv::COLOR_BGR2GRAY);

        if (frame_idx % det_skip == 0) {
            // Detection + Kalman update
            auto dets = detector.detect(frame);
            tracker.update(dets);
        } else if (!prev_gray.empty()) {
            // KLT propagation
            tracker.propagate(prev_gray, curr_gray);
        }

        total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - t0).count();

        // Draw
        for (const auto& [id, trk] : tracker.tracks()) {
            if (!trk.confirmed || !trk.smooth_init) continue;
            cv::Rect2f b = trk.smooth_bbox;
            cv::Rect   bi((int)b.x, (int)b.y, (int)b.width, (int)b.height);
            bi &= cv::Rect(0, 0, W, H);
            if (bi.area() <= 0) continue;

            cv::Scalar col = idColor(id);
            cv::rectangle(frame, bi, col, 2);
            std::string label = "tank #" + std::to_string(id);
            cv::rectangle(frame, {bi.x, bi.y-20, (int)label.size()*10, 20}, col, cv::FILLED);
            cv::putText(frame, label, {bi.x+2, bi.y-5},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1, cv::LINE_AA);
        }

        // FPS overlay
        if (frame_idx > 0) {
            double avg_ms = (total_ns / frame_idx) / 1e6;
            std::string fps_str = "FPS: " + std::to_string((int)(1000/avg_ms));
            cv::putText(frame, fps_str, {10, 30},
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, {0,255,0}, 2);
        }

        if (writer.isOpened()) writer.write(frame);

        if (++frame_idx % 50 == 0) {
            double avg_ms = (total_ns / frame_idx) / 1e6;
            printf("Frame %d/%d  avg=%.1f ms  FPS=%.0f\n",
                   frame_idx, total_frames, avg_ms, 1000/avg_ms);
        }

        cv::swap(curr_gray, prev_gray);
    }

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    printf("\n=== Done ===\n");
    printf("Frames: %d  Time: %.1f s  FPS: %.1f\n",
           frame_idx, elapsed, frame_idx/elapsed);
    if (!output_path.empty())
        printf("Output: %s\n", output_path.c_str());

    return 0;
}
