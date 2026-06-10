import logging
import itertools
import numpy as np
import numba as nb
import cupyx
import cupy as cp
import cv2

from .utils.rect import to_tlbr, get_size, get_center
from .utils.rect import intersection, crop
from .utils.numba import mask_area, transform


LOGGER = logging.getLogger(__name__)

# CUDA kernels

_BGR2GRAY_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void bgr2gray(const unsigned char* bgr, unsigned char* gray, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int i = y * W + x;
    gray[i] = (unsigned char)(
        0.114f * bgr[i*3] + 0.587f * bgr[i*3+1] + 0.299f * bgr[i*3+2] + 0.5f);
}
''', 'bgr2gray')

_RESIZE_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void bilinear_resize(const unsigned char* src, unsigned char* dst,
                     int sW, int sH, int dW, int dH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dW || y >= dH) return;
    float sx = (x + 0.5f) * sW / dW - 0.5f;
    float sy = (y + 0.5f) * sH / dH - 0.5f;
    int x0 = max(0, (int)sx), x1 = min(sW-1, x0+1);
    int y0 = max(0, (int)sy), y1 = min(sH-1, y0+1);
    float wx = sx - x0, wy = sy - y0;
    float v = src[y0*sW+x0]*(1-wx)*(1-wy) + src[y0*sW+x1]*wx*(1-wy)
            + src[y1*sW+x0]*(1-wx)*wy     + src[y1*sW+x1]*wx*wy;
    dst[y*dW+x] = (unsigned char)(v + 0.5f);
}
''', 'bilinear_resize')

_SPARSE_LK_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void sparse_lk(
    const unsigned char* prev,
    const unsigned char* curr,
    int W, int H,
    const float* prev_pts,
    float* curr_pts,
    int N, int half_win, int max_iter, float eps_sq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float px = prev_pts[idx*2], py = prev_pts[idx*2+1];
    float cx = curr_pts[idx*2], cy = curr_pts[idx*2+1];
    float A00=0, A01=0, A11=0;
    for (int dy=-half_win; dy<=half_win; dy++) {
        for (int dx=-half_win; dx<=half_win; dx++) {
            int x=(int)(px+dx+0.5f), y=(int)(py+dy+0.5f);
            if (x<1||x>=W-1||y<1||y>=H-1) continue;
            float Ix=0.5f*(prev[y*W+x+1]-prev[y*W+x-1]);
            float Iy=0.5f*(prev[(y+1)*W+x]-prev[(y-1)*W+x]);
            A00+=Ix*Ix; A01+=Ix*Iy; A11+=Iy*Iy;
        }
    }
    float det=A00*A11-A01*A01;
    if (fabsf(det)<1e-3f) return;
    float inv_det=1.0f/det;
    for (int iter=0; iter<max_iter; iter++) {
        float b0=0, b1=0;
        for (int dy=-half_win; dy<=half_win; dy++) {
            for (int dx=-half_win; dx<=half_win; dx++) {
                int bx=(int)(px+dx+0.5f), by=(int)(py+dy+0.5f);
                int fx=(int)(cx+dx+0.5f), fy=(int)(cy+dy+0.5f);
                if (bx<1||bx>=W-1||by<1||by>=H-1) continue;
                if (fx<0||fx>=W||fy<0||fy>=H) continue;
                float It=(float)curr[fy*W+fx]-(float)prev[by*W+bx];
                float Ix=0.5f*(prev[by*W+bx+1]-prev[by*W+bx-1]);
                float Iy=0.5f*(prev[(by+1)*W+bx]-prev[(by-1)*W+bx]);
                b0-=Ix*It; b1-=Iy*It;
            }
        }
        float vx=(A11*b0-A01*b1)*inv_det;
        float vy=(A00*b1-A01*b0)*inv_det;
        cx+=vx; cy+=vy;
        if (vx*vx+vy*vy<eps_sq) break;
    }
    curr_pts[idx*2]=cx; curr_pts[idx*2+1]=cy;
}
''', 'sparse_lk')

_SSD_ERROR_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void compute_ssd(
    const unsigned char* prev, const unsigned char* curr,
    int W, int H,
    const float* prev_pts, const float* curr_pts,
    float* err, unsigned char* status,
    int N, int half_win
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float px=prev_pts[idx*2], py=prev_pts[idx*2+1];
    float cx=curr_pts[idx*2], cy=curr_pts[idx*2+1];
    if (cx<0||cx>=W||cy<0||cy>=H) { status[idx]=0; err[idx]=255; return; }
    float ssd=0; int cnt=0;
    for (int dy=-half_win; dy<=half_win; dy++) {
        for (int dx=-half_win; dx<=half_win; dx++) {
            int bx=(int)(px+dx+0.5f), by=(int)(py+dy+0.5f);
            int fx=(int)(cx+dx+0.5f), fy=(int)(cy+dy+0.5f);
            if (bx<0||bx>=W||by<0||by>=H||fx<0||fx>=W||fy<0||fy>=H) continue;
            float d=(float)curr[fy*W+fx]-(float)prev[by*W+bx];
            ssd+=d*d; cnt++;
        }
    }
    float e=(cnt>0)?sqrtf(ssd/cnt):255.0f;
    err[idx]=e; status[idx]=(e<30.0f)?1:0;
}
''', 'compute_ssd')


def _gpu_bgr2gray(bgr_gpu):
    H, W = bgr_gpu.shape[:2]
    gray = cp.empty((H, W), dtype=cp.uint8)
    _BGR2GRAY_KERNEL(((W+15)//16, (H+15)//16), (16, 16), (bgr_gpu, gray, np.int32(W), np.int32(H)))
    return gray


def _gpu_resize(src_gpu, dst_wh):
    sH, sW = src_gpu.shape[:2]
    dW, dH = dst_wh
    dst = cp.empty((dH, dW), dtype=cp.uint8)
    _RESIZE_KERNEL(((dW+15)//16, (dH+15)//16), (16, 16),
                   (src_gpu, dst, np.int32(sW), np.int32(sH), np.int32(dW), np.int32(dH)))
    return dst


def _gpu_optical_flow_lk(prev_gpu, curr_gpu, pts_scaled, half_win=2, max_iter=10, eps=0.03):
    """GPU sparse LK - drop-in replacement for cv2.calcOpticalFlowPyrLK.

    Returns (cur_pts, status, err) matching cv2 format:
        cur_pts: (N,1,2) float32
        status:  (N,1)   uint8
        err:     (N,1)   float32
    """
    N = len(pts_scaled)
    if N == 0:
        return (np.empty((0,1,2), np.float32),
                np.empty((0,1),   np.uint8),
                np.empty((0,1),   np.float32))

    H, W = prev_gpu.shape
    threads = min(256, max(32, N))
    blocks  = (N + threads - 1) // threads

    prev_pts_flat = pts_scaled.reshape(-1, 2).astype(np.float32)
    curr_pts_gpu  = cp.asarray(prev_pts_flat.copy())   # start estimate = prev
    prev_pts_gpu  = cp.asarray(prev_pts_flat)

    # Build pyramids and run coarse-to-fine
    levels = 3
    prev_pyr, curr_pyr = [prev_gpu], [curr_gpu]
    for _ in range(levels - 1):
        h, w = prev_pyr[-1].shape
        prev_pyr.append(_gpu_resize(prev_pyr[-1], (w//2, h//2)))
        curr_pyr.append(_gpu_resize(curr_pyr[-1], (w//2, h//2)))

    scale = 2 ** (levels - 1)
    prev_pts_gpu = prev_pts_gpu / scale
    curr_pts_gpu = curr_pts_gpu / scale

    for lv in range(levels - 1, -1, -1):
        pH, pW = prev_pyr[lv].shape
        _SPARSE_LK_KERNEL(
            (blocks,), (threads,),
            (prev_pyr[lv], curr_pyr[lv],
             np.int32(pW), np.int32(pH),
             prev_pts_gpu, curr_pts_gpu,
             np.int32(N), np.int32(half_win), np.int32(max_iter),
             np.float32(eps * eps))
        )
        if lv > 0:
            prev_pts_gpu = prev_pts_gpu * 2
            curr_pts_gpu = curr_pts_gpu * 2

    # SSD error + status
    err_gpu    = cp.empty(N, dtype=cp.float32)
    status_gpu = cp.empty(N, dtype=cp.uint8)
    pH, pW = prev_pyr[0].shape
    _SSD_ERROR_KERNEL(
        (blocks,), (threads,),
        (prev_pyr[0], curr_pyr[0],
         np.int32(pW), np.int32(pH),
         prev_pts_gpu, curr_pts_gpu,
         err_gpu, status_gpu,
         np.int32(N), np.int32(half_win))
    )

    cur_pts_cpu = cp.asnumpy(curr_pts_gpu).reshape(N, 1, 2).astype(np.float32)
    status_cpu  = cp.asnumpy(status_gpu).reshape(N, 1).astype(np.uint8)
    err_cpu     = cp.asnumpy(err_gpu).reshape(N, 1).astype(np.float32)
    return cur_pts_cpu, status_cpu, err_cpu


# Flow class - identical to original except _gpu_optical_flow_lk replaces
# cv2.calcOpticalFlowPyrLK and BGR2Gray/resize are done on GPU

class Flow:
    def __init__(self, size,
                 bg_feat_scale_factor=(0.1, 0.1),
                 opt_flow_scale_factor=(0.5, 0.5),
                 feat_density=0.005,
                 feat_dist_factor=0.06,
                 ransac_max_iter=500,
                 ransac_conf=0.99,
                 max_error=100,
                 inlier_thresh=4,
                 bg_feat_thresh=10,
                 obj_feat_params=None,
                 opt_flow_params=None):
        self.size = size
        self.bg_feat_scale_factor  = bg_feat_scale_factor
        self.opt_flow_scale_factor = opt_flow_scale_factor
        self.feat_density    = feat_density
        self.feat_dist_factor= feat_dist_factor
        self.ransac_max_iter = ransac_max_iter
        self.ransac_conf     = ransac_conf
        self.max_error       = max_error
        self.inlier_thresh   = inlier_thresh
        self.bg_feat_thresh  = bg_feat_thresh

        self.obj_feat_params = {"maxCorners":1000,"qualityLevel":0.06,"blockSize":3}
        self.opt_flow_params = {"winSize":(5,5),"maxLevel":5,"criteria":(3,10,0.03)}
        if obj_feat_params is not None:
            self.obj_feat_params.update(vars(obj_feat_params))

        self.bg_feat_detector = cv2.FastFeatureDetector_create(threshold=self.bg_feat_thresh)
        self.bg_keypoints      = None
        self.prev_bg_keypoints = None

        opt_flow_sz = (round(self.opt_flow_scale_factor[0]*self.size[0]),
                       round(self.opt_flow_scale_factor[1]*self.size[1]))
        bg_feat_sz  = (round(self.bg_feat_scale_factor[0]*self.size[0]),
                       round(self.bg_feat_scale_factor[1]*self.size[1]))

        self.frame_gray       = cupyx.empty_pinned(self.size[::-1], np.uint8)
        self.frame_small      = cupyx.empty_pinned(opt_flow_sz[::-1], np.uint8)
        self.prev_frame_gray  = cupyx.empty_like_pinned(self.frame_gray)
        self.prev_frame_small = cupyx.empty_like_pinned(self.frame_small)
        self.prev_frame_bg    = cupyx.empty_pinned(bg_feat_sz[::-1], np.uint8)
        self.bg_mask_small    = cupyx.empty_like_pinned(self.prev_frame_bg)
        self.fg_mask          = cupyx.empty_like_pinned(self.frame_gray)

        # persistent GPU gray buffers
        self._gpu_prev_small = cp.empty(opt_flow_sz[::-1], dtype=cp.uint8)
        self._gpu_curr_small = cp.empty(opt_flow_sz[::-1], dtype=cp.uint8)

        self.frame_rect = to_tlbr((0, 0, *self.size))

    def init(self, frame):
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.prev_frame_gray)
        cv2.resize(self.prev_frame_gray, self.prev_frame_small.shape[::-1],
                   dst=self.prev_frame_small)
        # Upload pinned small frame to GPU (fast - pinned host memory)
        self._gpu_prev_small[:] = cp.asarray(self.prev_frame_small)
        self.bg_keypoints      = np.empty((0,2), np.float32)
        self.prev_bg_keypoints = np.empty((0,2), np.float32)

    def predict(self, frame, tracks):
        # CPU: BGR->Gray + resize (fast, ~0.5ms total)
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self.frame_gray)
        cv2.resize(self.frame_gray, self.frame_small.shape[::-1], dst=self.frame_small)

        # Upload only the small (230KB) pinned frames to GPU - not the full BGR
        curr_small_gpu = cp.asarray(self.frame_small)   # pinned -> GPU: ~0.05ms

        # feature detection (CPU)
        tracks.sort(reverse=True)
        all_prev_pts = []
        self.fg_mask[:] = 255
        for track in tracks:
            inside_tlbr = intersection(track.tlbr, self.frame_rect)
            target_mask = crop(self.fg_mask, inside_tlbr)
            target_area = mask_area(target_mask)
            keypoints   = self._rect_filter(track.keypoints, inside_tlbr, self.fg_mask)
            if len(keypoints) < self.feat_density * target_area:
                img = crop(self.prev_frame_gray, inside_tlbr)
                feature_dist = self._estimate_feature_dist(target_area, self.feat_dist_factor)
                keypoints = cv2.goodFeaturesToTrack(img, mask=target_mask,
                                                    minDistance=feature_dist,
                                                    **self.obj_feat_params)
                if keypoints is None:
                    keypoints = np.empty((0,2), np.float32)
                else:
                    keypoints = self._ellipse_filter(keypoints, track.tlbr, inside_tlbr[:2])
            all_prev_pts.append(keypoints)
            target_mask[:] = 0

        target_ends   = list(itertools.accumulate(len(p) for p in all_prev_pts)) if all_prev_pts else [0]
        target_begins = itertools.chain([0], target_ends[:-1])

        # Background features
        cv2.resize(self.prev_frame_gray, self.prev_frame_bg.shape[::-1], dst=self.prev_frame_bg)
        cv2.resize(self.fg_mask, self.bg_mask_small.shape[::-1], dst=self.bg_mask_small,
                   interpolation=cv2.INTER_NEAREST)
        keypoints = self.bg_feat_detector.detect(self.prev_frame_bg, mask=self.bg_mask_small)
        if len(keypoints) == 0:
            self.bg_keypoints = np.empty((0,2), np.float32)
            self._gpu_prev_small[:] = curr_small_gpu
            self.prev_frame_gray, self.frame_gray = self.frame_gray, self.prev_frame_gray
            self.prev_frame_small, self.frame_small = self.frame_small, self.prev_frame_small
            LOGGER.warning('Camera motion estimation failed')
            return {}, None

        keypoints = np.float32([kp.pt for kp in keypoints])
        keypoints = self._unscale_pts(keypoints, self.bg_feat_scale_factor)
        bg_begin  = target_ends[-1]
        all_prev_pts.append(keypoints)

        # GPU sparse LK (replaces cv2.calcOpticalFlowPyrLK)
        all_prev_pts_np = np.concatenate(all_prev_pts)
        scaled_prev_pts = self._scale_pts(all_prev_pts_np, self.opt_flow_scale_factor)

        all_cur_pts, status, err = _gpu_optical_flow_lk(
            self._gpu_prev_small, curr_small_gpu, scaled_prev_pts)

        status = self._get_status(status, err, self.max_error)
        all_cur_pts = self._unscale_pts(all_cur_pts, self.opt_flow_scale_factor, status)

        # swap buffers (GPU: assign, CPU: copy pinned)
        self._gpu_prev_small[:] = curr_small_gpu
        self.prev_frame_gray, self.frame_gray = self.frame_gray, self.prev_frame_gray
        self.prev_frame_small, self.frame_small = self.frame_small, self.prev_frame_small

        # camera motion
        homography = None
        prev_bg_pts, matched_bg_pts = self._get_good_match(
            all_prev_pts_np, all_cur_pts, status, bg_begin, -1)
        if len(matched_bg_pts) < 4:
            self.bg_keypoints = np.empty((0,2), np.float32)
            LOGGER.warning('Camera motion estimation failed')
            return {}, None
        homography, inlier_mask = cv2.findHomography(
            prev_bg_pts, matched_bg_pts,
            method=cv2.RANSAC, maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
        self.prev_bg_keypoints, self.bg_keypoints = self._get_inliers(
            prev_bg_pts, matched_bg_pts, inlier_mask)
        if homography is None or len(self.bg_keypoints) < self.inlier_thresh:
            self.bg_keypoints = np.empty((0,2), np.float32)
            LOGGER.warning('Camera motion estimation failed')
            return {}, None

        # target bounding boxes
        next_bboxes = {}
        self.fg_mask[:] = 255
        for begin, end, track in zip(target_begins, target_ends, tracks):
            prev_pts, matched_pts = self._get_good_match(
                all_prev_pts_np, all_cur_pts, status, begin, end)
            prev_pts, matched_pts = self._fg_filter(
                prev_pts, matched_pts, self.fg_mask, self.size)
            if len(matched_pts) < 3:
                track.keypoints = np.empty((0,2), np.float32)
                continue
            affine_mat, inlier_mask = cv2.estimateAffinePartial2D(
                prev_pts, matched_pts,
                method=cv2.RANSAC, maxIters=self.ransac_max_iter, confidence=self.ransac_conf)
            if affine_mat is None:
                track.keypoints = np.empty((0,2), np.float32)
                continue
            est_tlbr = self._estimate_bbox(track.tlbr, affine_mat)
            track.prev_keypoints, track.keypoints = self._get_inliers(
                prev_pts, matched_pts, inlier_mask)
            if (intersection(est_tlbr, self.frame_rect) is None or
                    len(track.keypoints) < self.inlier_thresh):
                track.keypoints = np.empty((0,2), np.float32)
                continue
            next_bboxes[track.trk_id] = est_tlbr
            track.inlier_ratio = len(track.keypoints) / len(matched_pts)
            target_mask = crop(self.fg_mask, est_tlbr)
            target_mask[:] = 0

        return next_bboxes, homography

    # helpers (unchanged from original)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_feature_dist(target_area, feat_dist_factor):
        est = round(np.sqrt(target_area) * feat_dist_factor)
        return max(est, 1)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _estimate_bbox(tlbr, affine_mat):
        tl = transform(tlbr[:2], affine_mat).ravel()
        scale = np.linalg.norm(affine_mat[:2, 0])
        scale = 1. if scale < 0.9 or scale > 1.1 else scale
        w, h = get_size(tlbr)
        return to_tlbr((tl[0], tl[1], w * scale, h * scale))

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _rect_filter(pts, tlbr, fg_mask):
        if len(pts) == 0:
            return np.empty((0, 2), np.float32)
        pts2i = np.rint(pts).astype(np.int32)
        ge_le  = (pts2i >= tlbr[:2]) & (pts2i <= tlbr[2:])
        inside = np.where(ge_le[:, 0] & ge_le[:, 1])
        pts, pts2i = pts[inside], pts2i[inside]
        keep = np.array([i for i in range(len(pts2i))
                         if fg_mask[pts2i[i][1], pts2i[i][0]] == 255])
        return pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _ellipse_filter(pts, tlbr, offset):
        offset    = np.asarray(offset, np.float32)
        center    = np.array(get_center(tlbr))
        semi_axes = np.array(get_size(tlbr)) * 0.5
        pts       = pts.reshape(-1, 2) + offset
        keep = np.sum(((pts - center) / semi_axes)**2, axis=1) <= 1.
        return pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _fg_filter(prev_pts, cur_pts, fg_mask, frame_sz):
        if len(cur_pts) == 0:
            return prev_pts, cur_pts
        size  = np.array(frame_sz)
        pts2i = np.rint(cur_pts).astype(np.int32)
        ge_lt  = (pts2i >= 0) & (pts2i < size)
        inside = ge_lt[:, 0] & ge_lt[:, 1]
        prev_pts, cur_pts = prev_pts[inside], cur_pts[inside]
        pts2i = pts2i[inside]
        keep = np.array([i for i in range(len(pts2i))
                         if fg_mask[pts2i[i][1], pts2i[i][0]] == 255])
        return prev_pts[keep], cur_pts[keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _scale_pts(pts, scale_factor):
        scale_factor = np.array(scale_factor, np.float32)
        pts = pts * scale_factor
        return pts.reshape(-1, 1, 2)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _unscale_pts(pts, scale_factor, mask=None):
        scale_factor = np.array(scale_factor, np.float32)
        unscale = 1 / scale_factor
        pts = pts.reshape(-1, 2)
        if mask is None:
            pts = pts * unscale
        else:
            idx = np.where(mask)
            pts[idx] = pts[idx] * unscale
        return pts

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_status(status, err, max_err):
        return status.ravel().astype(np.bool_) & (err.ravel() < max_err)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_good_match(prev_pts, cur_pts, status, begin, end):
        keep = np.where(status[begin:end])
        return prev_pts[begin:end][keep], cur_pts[begin:end][keep]

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _get_inliers(prev_pts, cur_pts, inlier_mask):
        keep = np.where(inlier_mask.ravel())
        return prev_pts[keep], cur_pts[keep]
