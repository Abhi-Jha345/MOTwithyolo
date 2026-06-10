// CUDA kernel: BGR uint8 -> letterboxed FP16 (CHW, normalized 0-1)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void letterbox_normalize_kernel(
    const uint8_t* __restrict__ src,  // BGR, HWC, src_h x src_w x 3
    float*         __restrict__ dst,  // RGB, CHW, dst_h x dst_w x 3 (FP32)
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale,
    int pad_w, int pad_h
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    int area = dst_w * dst_h;

    // Map dst pixel to src pixel (reverse letterbox)
    float src_x = (x - pad_w) / scale;
    float src_y = (y - pad_h) / scale;

    float r, g, b;
    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        r = g = b = 0.5f;   // padding = gray
    } else {
        // Bilinear interpolation
        int x0 = (int)src_x, y0 = (int)src_y;
        int x1 = min(x0+1, src_w-1), y1 = min(y0+1, src_h-1);
        float wx = src_x - x0, wy = src_y - y0;

        auto idx = [=](int row, int col) { return (row * src_w + col) * 3; };
        b = src[idx(y0,x0)+0]*(1-wx)*(1-wy) + src[idx(y0,x1)+0]*wx*(1-wy)
          + src[idx(y1,x0)+0]*(1-wx)*wy     + src[idx(y1,x1)+0]*wx*wy;
        g = src[idx(y0,x0)+1]*(1-wx)*(1-wy) + src[idx(y0,x1)+1]*wx*(1-wy)
          + src[idx(y1,x0)+1]*(1-wx)*wy     + src[idx(y1,x1)+1]*wx*wy;
        r = src[idx(y0,x0)+2]*(1-wx)*(1-wy) + src[idx(y0,x1)+2]*wx*(1-wy)
          + src[idx(y1,x0)+2]*(1-wx)*wy     + src[idx(y1,x1)+2]*wx*wy;
    }

    // Store as CHW FP32 (normalized 0-1), RGB order
    dst[0*area + y*dst_w + x] = r / 255.f;
    dst[1*area + y*dst_w + x] = g / 255.f;
    dst[2*area + y*dst_w + x] = b / 255.f;
}

extern "C" void preprocess_cuda(
    const uint8_t* bgr_host, void* dst_device_void,
    int src_w, int src_h, int dst_w, int dst_h,
    cudaStream_t stream
) {
    float* dst_device = reinterpret_cast<float*>(dst_device_void);
    // Upload frame to device
    uint8_t* d_src;
    size_t nbytes = (size_t)src_w * src_h * 3;
    cudaMallocAsync(&d_src, nbytes, stream);
    cudaMemcpyAsync(d_src, bgr_host, nbytes, cudaMemcpyHostToDevice, stream);

    float scale = fminf((float)dst_w/src_w, (float)dst_h/src_h);
    int new_w   = (int)(src_w * scale);
    int new_h   = (int)(src_h * scale);
    int pad_w   = (dst_w - new_w) / 2;
    int pad_h   = (dst_h - new_h) / 2;

    dim3 block(16, 16);
    dim3 grid((dst_w+15)/16, (dst_h+15)/16);
    letterbox_normalize_kernel<<<grid, block, 0, stream>>>(
        d_src, dst_device, src_w, src_h, dst_w, dst_h, scale, pad_w, pad_h);

    cudaFreeAsync(d_src, stream);
}
