"""
VigilantVision - Feature Extraction Pipeline
==============================================
Runs YOLOv8-Pose (person + skeletal keypoints) and YOLOv8 (item detection)
over UCF-Crime Shoplifting + Normal clips, computes per-frame geometric
features per tracked person, and labels each frame using the temporal
annotation file (theft window) or Normal-video negative label.

Output: a single CSV with one row per (video, frame, person_track_id)
ready to feed into a classifier (logistic regression / small MLP).

Usage:
    python extract_features.py \
        --dataset_dir ./dataset \
        --annotation_file Temporal_Anomaly_Annotation.txt \
        --output_csv features.csv \
        --frame_stride 3

Requires: pip install ultralytics opencv-python --break-system-packages
"""

import os
import csv
import argparse
import numpy as np
from ultralytics import YOLO

# COCO keypoint indices (17-point skeleton)
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_WRIST, RIGHT_WRIST = 9, 10

# COCO classes we treat as "holdable / product-like" items.
# Curated to reduce noise from irrelevant classes (cars, traffic lights, etc).
ITEM_CLASS_NAMES = {
    "backpack", "umbrella", "handbag", "suitcase", "bottle", "wine glass",
    "cup", "banana", "apple", "sandwich", "orange", "book", "vase",
    "scissors", "teddy bear", "toothbrush", "cell phone", "remote",
    "keyboard", "mouse", "laptop", "bowl", "clock",
}


def parse_annotations(path):
    """Returns dict: filename -> {'label': str, 'segments': [(s,e), ...]}"""
    entries = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            filename, label, s1, e1, s2, e2 = parts[:6]
            segs = [(int(s1), int(e1)), (int(s2), int(e2))]
            entries[filename] = {"label": label, "segments": segs}
    return entries


def frame_is_positive(frame_idx, segments):
    for s, e in segments:
        if s == -1:
            continue
        if s <= frame_idx <= e:
            return True
    return False


def find_local_videos(dataset_dir):
    found = {}
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(".mp4"):
                found[f] = os.path.join(root, f)
    return found


def dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def process_video(video_path, video_label, segments, pose_model, item_model,
                   frame_stride, writer, video_name):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [skip] could not open {video_path}")
        return 0

    frame_idx = 0
    rows_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        # --- Person detection + pose + tracking ---
        pose_results = pose_model.track(
            frame, persist=True, verbose=False, classes=[0]  # class 0 = person
        )

        # --- Item detection (no tracking needed, just per-frame boxes) ---
        item_results = item_model.predict(frame, verbose=False)
        item_boxes = []
        if len(item_results) > 0 and item_results[0].boxes is not None:
            names = item_results[0].names
            for box in item_results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = names.get(cls_id, "")
                if cls_name in ITEM_CLASS_NAMES:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    item_boxes.append((cx, cy))

        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            kpts_all = pose_results[0].keypoints.xy.cpu().numpy()  # (N, 17, 2)
            boxes = pose_results[0].boxes
            track_ids = (
                boxes.id.cpu().numpy().astype(int)
                if boxes.id is not None
                else [-1] * len(kpts_all)
            )
            box_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else None

            for person_idx, kpts in enumerate(kpts_all):
                track_id = int(track_ids[person_idx])

                ls, rs = kpts[LEFT_SHOULDER], kpts[RIGHT_SHOULDER]
                lh, rh = kpts[LEFT_HIP], kpts[RIGHT_HIP]
                lw, rw = kpts[LEFT_WRIST], kpts[RIGHT_WRIST]

                # Skip if key joints weren't detected (all zeros)
                if np.all(ls == 0) or np.all(lh == 0):
                    continue

                torso_center = (
                    (ls[0] + rs[0] + lh[0] + rh[0]) / 4,
                    (ls[1] + rs[1] + lh[1] + rh[1]) / 4,
                )

                # Person bbox height for scale normalization
                if box_xyxy is not None:
                    x1, y1, x2, y2 = box_xyxy[person_idx]
                    person_height = max(y2 - y1, 1.0)
                else:
                    person_height = 1.0

                wrist_to_torso = min(
                    dist(lw, torso_center) if not np.all(lw == 0) else 1e6,
                    dist(rw, torso_center) if not np.all(rw == 0) else 1e6,
                ) / person_height

                if item_boxes:
                    wrist_to_item = min(
                        min(dist(lw, ib) for ib in item_boxes) if not np.all(lw == 0) else 1e6,
                        min(dist(rw, ib) for ib in item_boxes) if not np.all(rw == 0) else 1e6,
                    ) / person_height
                else:
                    wrist_to_item = 1e6  # no items detected this frame

                num_items_nearby = sum(
                    1 for ib in item_boxes
                    if dist(lw, ib) / person_height < 1.5 or dist(rw, ib) / person_height < 1.5
                )

                label = 1 if (video_label == "Shoplifting" and frame_is_positive(frame_idx, segments)) else 0

                writer.writerow([
                    video_name, frame_idx, track_id,
                    round(wrist_to_torso, 4), round(wrist_to_item, 4),
                    num_items_nearby, label,
                ])
                rows_written += 1

        frame_idx += 1

    cap.release()
    return rows_written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--annotation_file", default="Temporal_Anomaly_Annotation.txt")
    parser.add_argument("--output_csv", default="features.csv")
    parser.add_argument("--frame_stride", type=int, default=3,
                         help="Process every Nth frame (higher = faster, less data)")
    parser.add_argument("--skip_unannotated_shoplifting", action="store_true", default=True)
    args = parser.parse_args()

    print("Loading models (will auto-download pretrained weights on first run)...")
    pose_model = YOLO("yolov8n-pose.pt")
    item_model = YOLO("yolov8n.pt")

    annotations = parse_annotations(args.annotation_file)
    local_videos = find_local_videos(args.dataset_dir)
    print(f"Found {len(local_videos)} local videos.")

    to_process = []
    for fname, path in local_videos.items():
        if fname in annotations:
            entry = annotations[fname]
            to_process.append((fname, path, entry["label"], entry["segments"]))
        elif "Shoplifting" in path and args.skip_unannotated_shoplifting:
            print(f"  [skip - no frame annotation] {fname}")
        elif "Normal" in path or "normal" in path.lower():
            to_process.append((fname, path, "Normal", [(-1, -1), (-1, -1)]))

    print(f"Will process {len(to_process)} videos.\n")

    total_rows = 0
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_name", "frame_idx", "track_id",
            "wrist_to_torso_norm", "wrist_to_item_norm",
            "num_items_nearby", "label",
        ])

        for i, (fname, path, label, segments) in enumerate(to_process):
            print(f"[{i+1}/{len(to_process)}] {fname} ({label})...")
            rows = process_video(
                path, label, segments, pose_model, item_model,
                args.frame_stride, writer, fname,
            )
            total_rows += rows
            print(f"    -> {rows} rows")

    print(f"\nDone. Wrote {total_rows} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
