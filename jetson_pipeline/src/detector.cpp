#include "detector.hpp"
#include <fstream>
#include <stdexcept>
#include <NvInferPlugin.h>

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}

Detector::Detector(const std::string& engine_path, float conf, float iou, int sz)
    : conf_thresh_(conf), iou_thresh_(iou), img_size_(sz)
{
    initLibNvInferPlugins(&logger_, "");
    loadEngine(engine_path);
    allocBuffers();
    cudaStreamCreate(&stream_);
}

Detector::~Detector() {
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_)   cudaStreamDestroy(stream_);
}

void Detector::loadEngine(const std::string& path) {
    auto data = readFile(path);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
    if (!engine_) throw std::runtime_error("Failed to deserialize engine");
    context_.reset(engine_->createExecutionContext());
}

void Detector::allocBuffers() {
    // Input/Output are FP32 (FP16 is internal compute only)
    size_t in_bytes  = 1LL * 3 * img_size_ * img_size_ * sizeof(float);
    size_t out_bytes = 1LL * out_count_ * 6 * sizeof(float);
    cudaMalloc(&d_input_,  in_bytes);
    cudaMalloc(&d_output_, out_bytes);

    // Set input shape and tell TRT where the buffers live
    nvinfer1::Dims4 in_dims{1, 3, img_size_, img_size_};
    context_->setInputShape("images", in_dims);
    context_->setTensorAddress("images",  d_input_);
    context_->setTensorAddress("output0", d_output_);
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    int src_h = frame.rows, src_w = frame.cols;

    // Preprocess on GPU (letterbox + normalize FP16)
    preprocess_cuda(frame.data, d_input_,
                    src_w, src_h, img_size_, img_size_, stream_);

    // TRT inference (async)
    context_->enqueueV3(stream_);

    // Copy FP32 output to host
    int n_floats = out_count_ * 6;
    std::vector<float> h_out(n_floats);
    cudaMemcpyAsync(h_out.data(), d_output_,
                    n_floats * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    float scale = std::min((float)img_size_ / src_w, (float)img_size_ / src_h);
    float pad_w = (img_size_ - src_w * scale) / 2.f;
    float pad_h = (img_size_ - src_h * scale) / 2.f;

    std::vector<Detection> dets;
    for (int i = 0; i < out_count_; i++) {
        const float* row = h_out.data() + i * 6;
        float conf = row[4];
        if (conf < conf_thresh_) continue;

        float x1 = (row[0] - pad_w) / scale;
        float y1 = (row[1] - pad_h) / scale;
        float x2 = (row[2] - pad_w) / scale;
        float y2 = (row[3] - pad_h) / scale;

        // Clamp to frame
        x1 = std::max(0.f, std::min(x1, (float)src_w));
        y1 = std::max(0.f, std::min(y1, (float)src_h));
        x2 = std::max(0.f, std::min(x2, (float)src_w));
        y2 = std::max(0.f, std::min(y2, (float)src_h));
        if (x2 <= x1 || y2 <= y1) continue;

        dets.push_back({x1, y1, x2, y2, conf, (int)row[5]});
    }
    return dets;
}
