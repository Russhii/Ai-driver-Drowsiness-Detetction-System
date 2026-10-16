"""
evaluate_dataset.py
===================
Offline evaluation harness for Improvement 3 (Real-World Evaluation).

Runs the DrowsinessDetector against labelled video datasets such as:
  - NTHU-DDD  (National Tsing Hua University Drowsy Driving Dataset)
  - YawDD     (Yawning Detection Dataset)

Expected dataset directory layout
-----------------------------------
dataset_root/
  subject_001/
    video_drowsy.mp4
    video_alert.mp4
  subject_002/
    ...

Each *_drowsy.* video gets ground_truth="DROWSY", each *_alert.* gets "ALERT".
Adjust the `_infer_label()` function below if your dataset uses a different
naming convention or a separate CSV annotation file.

Usage
-----
python evaluate_dataset.py --dataset /path/to/NTHU-DDD --output results.csv

Output
------
Prints a per-video report and an aggregate confusion matrix, then saves
a CSV with per-video precision/recall/F1.
"""

import os
import cv2
import csv
import argparse
import time

from detector import DrowsinessDetector, ConfusionMatrix


def _infer_label(video_path: str) -> str:
    """
    Derive ground-truth label from filename.
    Override this for datasets with separate annotation files.
    """
    name = os.path.basename(video_path).lower()
    if "drowsy" in name or "fatigue" in name or "sleepy" in name:
        return "DROWSY"
    return "ALERT"


def evaluate_video(video_path: str, ground_truth: str,
                   max_frames: int = None) -> dict:
    """
    Run detector on one video file; return per-video metrics dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return None

    detector = DrowsinessDetector()
    frame_n  = 0
    t0       = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        if max_frames and frame_n > max_frames:
            break
        detector.process_frame(frame, ground_truth=ground_truth)

    cap.release()
    elapsed = time.time() - t0
    fps     = frame_n / (elapsed + 1e-6)

    cm = detector.eval_metrics
    result = {
        "video":     os.path.basename(video_path),
        "label":     ground_truth,
        "frames":    frame_n,
        "fps":       round(fps, 1),
        "TP":        cm.tp,
        "FP":        cm.fp,
        "FN":        cm.fn,
        "TN":        cm.tn,
        "precision": round(cm.precision, 3),
        "recall":    round(cm.recall, 3),
        "f1":        round(cm.f1, 3),
        "accuracy":  round(cm.accuracy, 3),
    }
    print(f"  {os.path.basename(video_path):<40} "
          f"P={cm.precision:.2f} R={cm.recall:.2f} "
          f"F1={cm.f1:.2f}  ({frame_n} frames @ {fps:.1f} fps)")
    return result


def find_videos(dataset_root: str):
    """Walk dataset directory; yield (path, label) tuples."""
    extensions = {".mp4", ".avi", ".mov", ".mkv"}
    for dirpath, _, files in os.walk(dataset_root):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in extensions:
                full = os.path.join(dirpath, fname)
                yield full, _infer_label(full)


def main():
    ap = argparse.ArgumentParser(description="Dataset evaluation harness")
    ap.add_argument("--dataset",    required=True,
                    help="Root directory of labelled video dataset")
    ap.add_argument("--output",     default="eval_results.csv",
                    help="Output CSV path")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Max frames per video (for quick smoke tests)")
    args = ap.parse_args()

    videos = list(find_videos(args.dataset))
    if not videos:
        print(f"[ERROR] No video files found under {args.dataset}")
        return

    print(f"\nEvaluating {len(videos)} videos in: {args.dataset}\n")
    print(f"  {'Video':<40} Precision Recall F1")
    print("  " + "-" * 60)

    results = []
    global_cm = ConfusionMatrix()

    for path, label in videos:
        r = evaluate_video(path, label, args.max_frames)
        if r:
            results.append(r)
            # Accumulate global counts
            global_cm.tp += r["TP"]
            global_cm.fp += r["FP"]
            global_cm.fn += r["FN"]
            global_cm.tn += r["TN"]

    # Aggregate report
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)
    print(global_cm.report())
    print("=" * 60)

    # Save CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(results)
        print(f"\nPer-video results saved to: {args.output}")


if __name__ == "__main__":
    main()
