from multiprocessing.pool import ThreadPool
import numpy as np
import numba as nb
import cv2

from . import models
from .utils import TRTInference
from .utils.rect import multi_crop


class TorchFeatureExtractor:
    """PyTorch-based ReID using a pretrained MobileNetV3-Small backbone.

    Drops the classifier head and uses the 576-dim penultimate features as
    appearance embeddings, L2-normalised. Cosine distance is used for matching.
    No TensorRT or extra libraries required - just torchvision.
    """

    FEATURE_DIM = 576        # MobileNetV3-Small avgpool output
    INPUT_SIZE  = (128, 256) # (W, H) - standard ReID crop size
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, device=None):
        import torch
        import torchvision.models as tvm

        self.feature_dim = self.FEATURE_DIM
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._torch  = torch

        backbone = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        # Remove classifier - keep features (avgpool output)
        self._model = torch.nn.Sequential(
            backbone.features,
            backbone.avgpool,
            torch.nn.Flatten(1),
        ).to(self._device).eval()

        self._pending_crops = None  # stored between extract_async and postprocess

    @property
    def metric(self):
        return 'cosine'

    def extract_async(self, frame, tlbrs):
        """Crop bounding boxes from frame and store for postprocess."""
        self._pending_crops = self._crop(frame, tlbrs)

    def postprocess(self):
        """Run inference on stored crops and return L2-normalised embeddings."""
        crops = self._pending_crops
        if crops is None or len(crops) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return self._infer(crops)

    def __call__(self, frame, tlbrs):
        self.extract_async(frame, tlbrs)
        return self.postprocess()

    def null_embeddings(self, detections):
        n = len(detections)
        if n == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        emb = np.ones((n, self.feature_dim), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb

    def _crop(self, frame, tlbrs):
        """Crop and preprocess bounding box regions from a BGR frame."""
        H, W = frame.shape[:2]
        crops = []
        for box in tlbrs:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                crops.append(np.zeros((3, *self.INPUT_SIZE[::-1]), dtype=np.float32))
                continue
            patch = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            patch = cv2.resize(patch, self.INPUT_SIZE).astype(np.float32) / 255.0
            patch = (patch - self.MEAN) / self.STD
            crops.append(patch.transpose(2, 0, 1))  # HWC -> CHW
        return np.stack(crops, axis=0)  # (N, 3, H, W)

    def _infer(self, crops):
        import torch
        with torch.no_grad():
            t = torch.from_numpy(crops).to(self._device)
            feats = self._model(t).cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return feats / norms


class NullFeatureExtractor:
    """IoU-only fallback: returns identical unit embeddings so the tracker
    relies solely on Kalman filter + IoU instead of appearance matching."""

    FEATURE_DIM = 512

    def __init__(self):
        self.feature_dim = self.FEATURE_DIM
        self._last_n = 0

    @property
    def metric(self):
        return 'euclidean'

    def extract_async(self, frame, tlbrs):
        self._last_n = len(tlbrs)

    def postprocess(self):
        return self._null(self._last_n)

    def null_embeddings(self, detections):
        return self._null(len(detections))

    def _null(self, n):
        if n == 0:
            return np.empty((0, self.feature_dim))
        emb = np.ones((n, self.feature_dim), dtype=np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb


class FeatureExtractor:
    def __init__(self, model='OSNet025', batch_size=16):
        """A feature extractor for ReID embeddings.

        Parameters
        ----------
        model : str, optional
            ReID model to use.
            Must be the name of a class that inherits `models.ReID`.
        batch_size : int, optional
            Batch size for inference.
        """
        self.model = models.ReID.get_model(model)
        assert batch_size >= 1
        self.batch_size = batch_size

        self.feature_dim = self.model.OUTPUT_LAYOUT
        self.backend = TRTInference(self.model, self.batch_size)
        self.inp_handle = self.backend.input.host.reshape(self.batch_size, *self.model.INPUT_SHAPE)
        self.pool = ThreadPool()

        self.embeddings = []
        self.last_num_features = 0

    def __del__(self):
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()

    def __call__(self, frame, tlbrs):
        """Extract feature embeddings from bounding boxes synchronously."""
        self.extract_async(frame, tlbrs)
        return self.postprocess()

    @property
    def metric(self):
        return self.model.METRIC

    def extract_async(self, frame, tlbrs):
        """Extract feature embeddings from bounding boxes asynchronously."""
        imgs = multi_crop(frame, tlbrs)
        self.embeddings, cur_imgs = [], []
        # pipeline inference and preprocessing the next batch in parallel
        for offset in range(0, len(imgs), self.batch_size):
            cur_imgs = imgs[offset:offset + self.batch_size]
            self.pool.starmap(self._preprocess, enumerate(cur_imgs))
            if offset > 0:
                embedding_out = self.backend.synchronize()[0]
                self.embeddings.append(embedding_out)
            self.backend.infer_async()
        self.last_num_features = len(cur_imgs)

    def postprocess(self):
        """Synchronizes, applies postprocessing, and returns a NxM matrix of N
        extracted embeddings with dimension M.
        This API should be called after `extract_async`.
        """
        if self.last_num_features == 0:
            return np.empty((0, self.feature_dim))

        embedding_out = self.backend.synchronize()[0][:self.last_num_features * self.feature_dim]
        self.embeddings.append(embedding_out)
        embeddings = np.concatenate(self.embeddings).reshape(-1, self.feature_dim)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def null_embeddings(self, detections):
        """Returns a NxM matrix of N identical embeddings with dimension M.
        This API effectively disables feature extraction.
        """
        embeddings = np.ones((len(detections), self.feature_dim))
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings

    def _preprocess(self, idx, img):
        img = cv2.resize(img, self.model.INPUT_SHAPE[:0:-1])
        self._normalize(img, self.inp_handle[idx])

    @staticmethod
    @nb.njit(fastmath=True, nogil=True, cache=True)
    def _normalize(img, out):
        # BGR to RGB
        rgb = img[..., ::-1]
        # HWC -> CHW
        chw = rgb.transpose(2, 0, 1)
        # Normalize using ImageNet's mean and std
        out[0, ...] = (chw[0, ...] / 255. - 0.485) / 0.229
        out[1, ...] = (chw[1, ...] / 255. - 0.456) / 0.224
        out[2, ...] = (chw[2, ...] / 255. - 0.406) / 0.225
