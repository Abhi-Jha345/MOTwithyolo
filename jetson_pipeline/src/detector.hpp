#pragma once
#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int   cls;
};

// CUDA letterbox + normalize kernel (defined in preprocess.cu)
extern "C" void preprocess_cuda(const uint8_t* bgr_host, void* dst_device,
                                int src_w, int src_h, int dst_w, int dst_h,
                                cudaStream_t stream);

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TRT] %s\n", msg);
    }
};

class Detector {
public:
    explicit Detector(const std::string& engine_path,
                      float conf_thresh = 0.35f,
                      float iou_thresh  = 0.25f,
                      int   img_size    = 640);
    ~Detector();

    // Run inference on a BGR frame; returns detections in original-frame coords
    std::vector<Detection> detect(const cv::Mat& frame);

    int imgSize()  const { return img_size_; }

private:
    void loadEngine(const std::string& path);
    void allocBuffers();

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>        engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>  context_;

    // GPU buffers
    void*  d_input_  = nullptr;   // FP16 (1,3,H,W)
    void*  d_output_ = nullptr;   // FP16 (1,300,6)
    uint8_t* d_img_  = nullptr;   // raw BGR (pinned via device)

    cudaStream_t stream_ = nullptr;

    float conf_thresh_;
    float iou_thresh_;
    int   img_size_;

    int out_count_ = 300;  // YOLO26n end-to-end max detections
};
