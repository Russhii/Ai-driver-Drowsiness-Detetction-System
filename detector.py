"""
detector.py  (Improved – v2.0)
================================
High-accuracy drowsiness detection engine using MediaPipe Face Mesh.

Improvements over v1.0
-----------------------
1. DNN Fallback  — OpenCV's res10_300x300_ssd_iter_140000 face detector
   replaces fragile Haar cascades.  Runs at ~25 fps on CPU; handles low
   lighting and off-axis poses that Haar cascades routinely miss.
   Model files are fetched from the OpenCV GitHub CDN on first run and
   cached in ~/.cache/drowsiness_dnn/.

2. Dynamic EAR Baseline  — Instead of a fixed EAR threshold (0.21), the
   system spends the first CALIBRATION_FRAMES frames learning each driver's
   personal open-eye EAR distribution.  The detection threshold is then set
   to  personal_mean - 2.5 * personal_std  (approx 2.5 sigma below normal),
   which adapts automatically to glasses, eye shape, and lighting.

3. Dataset Evaluation Support  — process_frame() now accepts an optional
   ground_truth label so that callers can accumulate TP/FP/FN/TN
   counts for offline evaluation against NTHU-DDD or YawDD datasets.

Detects (same as v1.0):
  - Eye Aspect Ratio (EAR)   -- Soukupova & Cech 6-point formula
  - Mouth Aspect Ratio (MAR) -- yawn detection via landmarks
  - Head Pose (Pitch/Yaw)    -- 3D solvePnP with 6 canonical face points

Uses MediaPipe's 468-landmark Face Mesh (GPU-accelerated, no external file).
Falls back to OpenCV DNN SSD detector if MediaPipe is unavailable.
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from collections import deque

# Try MediaPipe first, fall back to DNN
try:
    import mediapipe as mp
    _MP_AVAILABLE = hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")
    if not _MP_AVAILABLE:
        print("[WARN] mediapipe solutions API unavailable — falling back to OpenCV DNN.")
except ImportError:
    _MP_AVAILABLE = False
    print("[WARN] mediapipe not found — falling back to OpenCV DNN face detector.")
    print("       Run: pip install mediapipe  for much better accuracy.")


# MediaPipe landmark indices
_L_EYE = [362, 385, 387, 263, 373, 380]
_R_EYE = [33,  160, 158, 133, 153, 144]

_MOUTH_TOP    = [13, 312, 311, 310]
_MOUTH_BOTTOM = [14,  82,  81,  80]
_MOUTH_HORIZ  = [78, 308]

_POSE_LANDMARKS = [1, 152, 263, 33, 287, 57]


# DNN model helpers
_DNN_CACHE_DIR = os.path.expanduser("~/.cache/drowsiness_dnn")
_DNN_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
_DNN_MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)


def _ensure_dnn_model():
    """Download DNN model files to cache dir if not already present."""
    os.makedirs(_DNN_CACHE_DIR, exist_ok=True)
    proto_path = os.path.join(_DNN_CACHE_DIR, "deploy.prototxt")
    model_path = os.path.join(_DNN_CACHE_DIR, "res10_300x300_ssd.caffemodel")
    if not os.path.exists(proto_path):
        print("[DNN] Downloading face detector prototxt...")
        urllib.request.urlretrieve(_DNN_PROTO_URL, proto_path)
    if not os.path.exists(model_path):
        print("[DNN] Downloading face detector model (~2.7 MB)...")
        urllib.request.urlretrieve(_DNN_MODEL_URL, model_path)
    return proto_path, model_path


def _load_dnn_detector():
    """Load the SSD MobileNet face detector. Returns cv2.dnn_Net."""
    proto, model = _ensure_dnn_model()
    return cv2.dnn.readNetFromCaffe(proto, model)


def _dnn_detect_faces(net, frame, conf_threshold=0.65):
    """
    Run DNN face detection on a BGR frame.
    Returns list of (x, y, w, h) bounding boxes sorted by area descending.
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False,
    )
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < conf_threshold:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))
        bw, bh = x2 - x1, y2 - y1
        if bw > 0 and bh > 0:
            boxes.append((x1, y1, bw, bh))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes


def _ear(landmarks, indices, w, h):
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
        dtype=np.float32
    )
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _mar(landmarks, w, h):
    top_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in _MOUTH_TOP],
        dtype=np.float32
    )
    bot_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in _MOUTH_BOTTOM],
        dtype=np.float32
    )
    vert  = np.mean([np.linalg.norm(t - b) for t, b in zip(top_pts, bot_pts)])
    l = np.array([landmarks[_MOUTH_HORIZ[0]].x * w,
                  landmarks[_MOUTH_HORIZ[0]].y * h], dtype=np.float32)
    r = np.array([landmarks[_MOUTH_HORIZ[1]].x * w,
                  landmarks[_MOUTH_HORIZ[1]].y * h], dtype=np.float32)
    return float(vert / (np.linalg.norm(l - r) + 1e-6))


def _head_pitch(landmarks, w, h):
    model_pts = np.array([
        [0.0,    0.0,    0.0],
        [0.0,  -330.0, -65.0],
        [-225.0, 170.0,-135.0],
        [225.0,  170.0,-135.0],
        [-150.0,-150.0,-125.0],
        [150.0, -150.0,-125.0],
    ], dtype=np.float64)
    image_pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in _POSE_LANDMARKS],
        dtype=np.float64
    )
    focal  = w
    center = (w / 2, h / 2)
    cam_mat = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0, 0, 1]], dtype=np.float64)
    success, rvec, _ = cv2.solvePnP(
        model_pts, image_pts, cam_mat, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    sy    = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    return float(np.degrees(np.arctan2(-rmat[2, 0], sy)))


# =====================================================================
# IMPROVEMENT 2 — Dynamic EAR Baseline Calibration
# =====================================================================
class EARCalibrator:
    """
    Personalised EAR threshold via first-N-frame statistics.

    Collects open-eye EAR samples during the first `n_frames` frames,
    then sets threshold = mean - sigma_factor * std.

    Replaces the hard-coded 0.21 constant from v1.0. Adapts to:
      - Glasses wearers (EAR is systematically lower due to frame reflection)
      - Small / partially obscured eyes
      - Variable ambient lighting shifting landmark confidence
    """

    DEFAULT_THRESHOLD = 0.21
    SIGMA_FACTOR      = 2.5

    def __init__(self, n_frames=450, sigma_factor=None):
        self._n       = n_frames
        self._sigma   = sigma_factor or self.SIGMA_FACTOR
        self._samples = deque(maxlen=n_frames)
        self._done    = False
        self._thresh  = self.DEFAULT_THRESHOLD

    def update(self, ear_value: float):
        if self._done:
            return
        if ear_value > 0.15:         # ignore obvious blinks during calibration
            self._samples.append(ear_value)
        if len(self._samples) >= self._n:
            arr = np.array(self._samples)
            self._thresh = float(np.mean(arr) - self._sigma * np.std(arr))
            self._thresh = max(0.10, min(0.28, self._thresh))  # safety clamp
            self._done   = True
            print(f"[EARCalibrator] Done. mean={np.mean(arr):.3f}, "
                  f"std={np.std(arr):.3f}, threshold={self._thresh:.3f}")

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def threshold(self) -> float:
        return self._thresh

    @property
    def progress(self) -> float:
        return min(1.0, len(self._samples) / self._n)


# =====================================================================
# IMPROVEMENT 3 — Dataset Evaluation / Confusion Matrix
# =====================================================================
class ConfusionMatrix:
    """
    Accumulates TP/FP/FN/TN for offline evaluation against
    labelled datasets such as NTHU-DDD or YawDD.

    Usage:
        for frame, label in dataset:
            detector.process_frame(frame, ground_truth=label)
        print(detector.eval_metrics.report())
    """

    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, predicted_state: str, ground_truth: str):
        pred_drowsy = predicted_state in ("DROWSY", "WARNING")
        gt_drowsy   = ground_truth == "DROWSY"
        if   pred_drowsy and     gt_drowsy: self.tp += 1
        elif pred_drowsy and not gt_drowsy: self.fp += 1
        elif not pred_drowsy and gt_drowsy: self.fn += 1
        else:                               self.tn += 1

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self):
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self):
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 0.0

    def report(self) -> str:
        return (
            f"Confusion Matrix\n"
            f"  TP={self.tp}  FP={self.fp}  FN={self.fn}  TN={self.tn}\n"
            f"  Precision : {self.precision:.3f}\n"
            f"  Recall    : {self.recall:.3f}\n"
            f"  F1        : {self.f1:.3f}\n"
            f"  Accuracy  : {self.accuracy:.3f}"
        )


class EyeState:
    def __init__(self, calibrator: EARCalibrator):
        self._cal        = calibrator
        self.left_ear    = 0.3
        self.right_ear   = 0.3
        self.ear         = 0.3
        self.both_closed = False
        self.left_open   = True
        self.right_open  = True

    @property
    def ear_threshold(self):
        return self._cal.threshold

    def update_from_landmarks(self, landmarks, w, h):
        self.left_ear  = _ear(landmarks, _L_EYE, w, h)
        self.right_ear = _ear(landmarks, _R_EYE, w, h)
        self.ear       = (self.left_ear + self.right_ear) / 2.0
        thresh = self.ear_threshold
        self.left_open   = self.left_ear  > thresh
        self.right_open  = self.right_ear > thresh
        self.both_closed = (not self.left_open) and (not self.right_open)

    def update_dnn(self, face_roi_gray, face_w, face_h):
        upper = face_roi_gray[:int(face_h * 0.55), :]
        if upper.size == 0:
            return
        lap    = cv2.Laplacian(upper, cv2.CV_64F)
        energy = np.var(lap)
        norm_energy = energy / (face_w * face_h + 1e-6)
        OPEN_ENERGY  = 0.40
        self.ear        = float(np.clip(norm_energy / (OPEN_ENERGY + 1e-6), 0, 1))
        self.left_open  = self.ear > 0.35
        self.right_open = self.ear > 0.35
        self.both_closed = (not self.left_open) and (not self.right_open)


class MouthState:
    MAR_THRESH = 0.60

    def __init__(self):
        self.yawning = False
        self.mor     = 0.0

    def update_from_landmarks(self, landmarks, w, h):
        self.mor     = _mar(landmarks, w, h)
        self.yawning = self.mor > self.MAR_THRESH

    def update_dnn(self, face_roi_gray, face_h):
        lower = face_roi_gray[int(face_h * 0.60):, :]
        if lower.size == 0:
            return
        sobel_y  = cv2.Sobel(lower, cv2.CV_64F, 0, 1, ksize=3)
        self.mor = float(np.clip(np.std(sobel_y) / 30.0, 0, 1))
        self.yawning = self.mor > 0.50


class HeadPose:
    PITCH_THRESH = 20.0

    def __init__(self):
        self.tilted   = False
        self.tilt_val = 0.0
        self._pitch   = 0.0
        self._prev_cy = None

    def update_from_landmarks(self, landmarks, w, h, frame_h):
        self._pitch   = _head_pitch(landmarks, w, h)
        self.tilt_val = float(np.clip(abs(self._pitch) / 45.0, 0.0, 1.0))
        self.tilted   = self._pitch > self.PITCH_THRESH

    def update_dnn(self, face_rect, frame_h):
        x, y, w, h   = face_rect
        cy            = y + h // 2
        normalized_y  = cy / (frame_h + 1e-6)
        if self._prev_cy is not None:
            delta = normalized_y - self._prev_cy
            self.tilt_val = float(np.clip(abs(delta) * 20, 0.0, 1.0))
        else:
            self.tilt_val = 0.0
        self._prev_cy = normalized_y
        self.tilted   = normalized_y > 0.72

    @property
    def pitch_deg(self):
        return self._pitch


class DrowsinessDetector:
    """
    Main detection class — uses MediaPipe Face Mesh for high accuracy.

    v2.0 improvements:
      * DNN SSD fallback replaces Haar cascades  (Improvement 1)
      * Dynamic per-driver EAR calibration       (Improvement 2)
      * Evaluation hook for dataset benchmarking (Improvement 3)
    """

    EYE_CLOSED_CONSEC_FRAMES = 18
    YAWN_CONSEC_FRAMES       = 20
    SCORE_DROWSY_THRESH      = 0.50
    SCORE_ALERT_THRESH       = 0.25

    def __init__(self):
        self.calibrator   = EARCalibrator(n_frames=450)
        self.eye_state    = EyeState(self.calibrator)
        self.mouth_state  = MouthState()
        self.head_pose    = HeadPose()
        self.eval_metrics = ConfusionMatrix()

        self._eye_closed_counter = 0
        self._yawn_counter       = 0
        self._no_face_counter    = 0

        self.drowsiness_score = 0.0
        self.state            = "ALERT"
        self.face_detected    = False

        if _MP_AVAILABLE:
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.55, min_tracking_confidence=0.55,
            )
            self._use_mp = True
            self._dnn_net = None
        else:
            self._use_mp = False
            try:
                print("[DNN] Loading SSD face detector (replacing Haar)...")
                self._dnn_net = _load_dnn_detector()
                print("[DNN] Face detector loaded OK.")
            except Exception as e:
                print(f"[DNN] Could not load DNN model ({e}).")
                self._dnn_net = None

        self._smooth_ear = 0.3
        self._smooth_mar = 0.0
        self._landmarks  = None

    def process_frame(self, frame, ground_truth: str = None):
        """
        Parameters
        ----------
        frame        : np.ndarray  BGR camera frame
        ground_truth : str or None  "DROWSY"/"ALERT" for dataset evaluation
        """
        if self._use_mp:
            self._process_mediapipe(frame)
        else:
            self._process_dnn(frame)
        if ground_truth is not None:
            self.eval_metrics.update(self.state, ground_truth)
        return self

    def _process_mediapipe(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mp_face_mesh.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_face_landmarks:
            self._no_face_counter += 1
            self.face_detected = False
            if self._no_face_counter > 30:
                self.state = "NO_FACE"
                self.drowsiness_score = 0.0
            return

        self.face_detected    = True
        self._no_face_counter = 0
        lm = results.multi_face_landmarks[0].landmark
        self._landmarks = lm

        self.eye_state.update_from_landmarks(lm, w, h)
        self.mouth_state.update_from_landmarks(lm, w, h)
        self.head_pose.update_from_landmarks(lm, w, h, h)

        if not self.calibrator.is_done:
            self.calibrator.update(self.eye_state.ear)

        alpha_m = 0.40
        self._smooth_ear = alpha_m * self.eye_state.ear   + (1 - alpha_m) * self._smooth_ear
        self._smooth_mar = alpha_m * self.mouth_state.mor + (1 - alpha_m) * self._smooth_mar

        if self.eye_state.both_closed:
            self._eye_closed_counter += 1
        else:
            self._eye_closed_counter = max(0, self._eye_closed_counter - 3)

        if self.mouth_state.yawning:
            self._yawn_counter += 1
        else:
            self._yawn_counter = max(0, self._yawn_counter - 1)

        eye_score  = min(self._eye_closed_counter / self.EYE_CLOSED_CONSEC_FRAMES, 1.0)
        yawn_score = min(self._yawn_counter / self.YAWN_CONSEC_FRAMES, 1.0)
        raw_score  = eye_score * 0.55 + yawn_score * 0.30 + self.head_pose.tilt_val * 0.15
        self.drowsiness_score = 0.30 * raw_score + 0.70 * self.drowsiness_score
        self._update_state()

    def _process_dnn(self, frame):
        h, w = frame.shape[:2]
        if self._dnn_net is None:
            self.face_detected = False
            return

        faces = _dnn_detect_faces(self._dnn_net, frame)
        if not faces:
            self._no_face_counter += 1
            self.face_detected = False
            if self._no_face_counter > 30:
                self.state = "NO_FACE"
                self.drowsiness_score = 0.0
            return

        fx, fy, fw, fh = faces[0]
        self.face_detected    = True
        self._no_face_counter = 0

        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[fy:fy+fh, fx:fx+fw]

        self.eye_state.update_dnn(face_roi, fw, fh)
        self.mouth_state.update_dnn(face_roi, fh)
        self.head_pose.update_dnn((fx, fy, fw, fh), h)

        if self.eye_state.both_closed:
            self._eye_closed_counter += 1
        else:
            self._eye_closed_counter = max(0, self._eye_closed_counter - 2)
        if self.mouth_state.yawning:
            self._yawn_counter += 1
        else:
            self._yawn_counter = max(0, self._yawn_counter - 1)

        eye_score  = min(self._eye_closed_counter / self.EYE_CLOSED_CONSEC_FRAMES, 1.0)
        yawn_score = min(self._yawn_counter / self.YAWN_CONSEC_FRAMES, 1.0)
        head_score = 1.0 if self.head_pose.tilted else 0.0
        raw_score  = eye_score * 0.60 + yawn_score * 0.25 + head_score * 0.15
        self.drowsiness_score = 0.35 * raw_score + 0.65 * self.drowsiness_score
        self._update_state()

    def _update_state(self):
        if   self.drowsiness_score >= self.SCORE_DROWSY_THRESH: self.state = "DROWSY"
        elif self.drowsiness_score >= self.SCORE_ALERT_THRESH:  self.state = "WARNING"
        else:                                                    self.state = "ALERT"

    @property
    def eye_closed_frames(self): return self._eye_closed_counter
    @property
    def yawn_frames(self):       return self._yawn_counter
    @property
    def landmarks(self):         return self._landmarks
    @property
    def smooth_ear(self):        return self._smooth_ear
    @property
    def smooth_mar(self):        return self._smooth_mar

    def reset(self):
        self._eye_closed_counter = 0
        self._yawn_counter       = 0
        self._no_face_counter    = 0
        self.drowsiness_score    = 0.0
        self.state               = "ALERT"
        self._smooth_ear         = 0.3
        self._smooth_mar         = 0.0
