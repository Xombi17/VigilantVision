"""
VigilantVision — Clip Preparation for 3D-CNN Training
======================================================
Trims UCF-Crime Shoplifting videos into short clips centered on annotated
theft windows, and samples random clips from Normal videos.

Use this BEFORE running train_clip_classifier.py.

Output:
    clips/
    ├── shoplifting_001.mp4     # Positive clips (theft windows)
    ├── shoplifting_002.mp4
    ├── ...
    ├── normal_001.mp4           # Negative clips (random Normal windows)
    ├── normal_002.mp4
    ├── ...
    └── clips_manifest.csv       # CSV: clip_path,label,source_video

Usage:
    python3 scripts/prepare_clips.py \
        --dataset_dir ./dataset \
        --annotation_file Temporal_Anomaly_Annotation.txt \
        --output_dir clips \
        --clip_duration 4.5 \
        --max_clips_per_video 5
"""

import os
import sys
import csv
import random
import argparse
import subprocess
import shutil

random.seed(42)

# ffmpeg is required for trimming
# Check it's available
if shutil.which("ffmpeg") is None:
    sys.exit("ffmpeg is required. Install it and add it to your PATH (e.g. using 'winget install ffmpeg').")
if shutil.which("ffprobe") is None:
    sys.exit("ffprobe is required. Install it and add it to your PATH (e.g. using 'winget install ffmpeg').")


def parse_annotations(path):
    """Returns dict: filename -> {'label': str, 'segments': [(s,e), ...]}"""
    entries = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            filename, label, s1, e1, s2, e2 = parts[:6]
            segs = [(int(s1), int(e1)), (int(s2), int(e2))]
            entries[filename] = {"label": label, "segments": segs}
    return entries


def find_local_videos(dataset_dir):
    """Returns dict: filename -> full_path"""
    found = {}
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith(".mp4"):
                found[f] = os.path.join(root, f)
    return found


def get_video_info(video_path):
    """Get video duration (seconds) and frame rate using ffprobe.
    Returns (duration_sec, fps) or (None, None) on failure.
    """
    try:
        # Duration
        dur_result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, timeout=30,
        )
        duration = float(dur_result.stdout.strip())

        # Frame rate (as fraction like "30000/1001")
        fps_result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, timeout=30,
        )
        fps_str = fps_result.stdout.strip()
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str) if fps_str else 30.0

        return duration, fps
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError, ZeroDivisionError):
        return None, None


def extract_clip(input_path, output_path, start_sec, duration_sec):
    """Extract a clip using ffmpeg (fast, no re-encode with copy)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration_sec),
        "-c", "copy",        # fast copy, no re-encode
        "-an",               # no audio
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--annotation_file", default="Temporal_Anomaly_Annotation.txt")
    parser.add_argument("--output_dir", default="clips")
    parser.add_argument("--clip_duration", type=float, default=4.5,
                        help="Duration of each clip in seconds")
    parser.add_argument("--max_clips_per_video", type=int, default=5,
                        help="Max clips to extract from a single video")
    args = parser.parse_args()

    # 1 --- Load annotations and find local videos
    annotations = parse_annotations(args.annotation_file)
    local_videos = find_local_videos(args.dataset_dir)
    print(f"Found {len(local_videos)} local videos.")

    # 2 --- Determine which videos to process
    # Positive: Shoplifting clips WITH frame-level annotations
    # Negative: Normal clips
    pos_videos = []    # (filename, path, segments)
    neg_videos = []    # (filename, path)

    for fname, path in local_videos.items():
        if fname in annotations:
            entry = annotations[fname]
            if entry["label"] == "Shoplifting":
                # Only use if it has non-trivial segments
                valid_segs = [(s, e) for s, e in entry["segments"] if s != -1]
                if valid_segs:
                    pos_videos.append((fname, path, valid_segs))
            elif entry["label"] == "Normal":
                neg_videos.append((fname, path))
        elif "Normal" in path or "normal" in path.lower():
            neg_videos.append((fname, path))

    print(f"Positive videos (annotated Shoplifting): {len(pos_videos)}")
    print(f"Negative videos (Normal): {len(neg_videos)}")

    # 3 --- Extract clips
    os.makedirs(args.output_dir, exist_ok=True)
    manifest_rows = []
    clip_idx = 0

    # --- Positive clips (centered on theft windows) ---
    print("\n=== Extracting Positive Clips ===")
    for fname, path, segments in pos_videos:
        duration, fps = get_video_info(path)
        if duration is None:
            print(f"  [skip] can't read video info: {fname}")
            continue
        print(f"  {fname}: {duration:.1f}s @ {fps:.2f}fps")

        clips_from_video = 0
        for seg_start, seg_end in segments:
            if clips_from_video >= args.max_clips_per_video:
                break

            seg_duration_sec = (seg_end - seg_start) / fps

            # Center the clip on the theft window
            window_mid = seg_start / fps + seg_duration_sec / 2
            clip_start = max(0, window_mid - args.clip_duration / 2)

            # Ensure clip fits within the video
            if clip_start + args.clip_duration > duration:
                clip_start = max(0, duration - args.clip_duration)

            if clip_start + args.clip_duration > duration:
                print(f"  [skip] clip too short: {fname}")
                continue

            clip_path = os.path.join(args.output_dir, f"shoplifting_{clip_idx:04d}.mp4")
            success = extract_clip(path, clip_path, clip_start, args.clip_duration)

            if success and os.path.getsize(clip_path) > 1024:
                # Store relative path so manifest works from any machine
                manifest_rows.append((clip_path, 1, fname))
                clip_idx += 1
                clips_from_video += 1
                print(f"  [{clip_idx}] {fname} @ {clip_start:.1f}s (theft window {seg_start}-{seg_end})")
            else:
                print(f"  [fail] {fname}")

    # --- Negative clips (random windows from Normal videos) ---
    print("\n=== Extracting Negative Clips ===")
    # Aim for roughly balanced classes (match positive count)
    target_negatives = clip_idx  # aim for balanced
    random.shuffle(neg_videos)

    for fname, path in neg_videos:
        if clip_idx >= target_negatives * 2:
            break  # generous upper limit

        duration, _ = get_video_info(path)
        if duration is None or duration < args.clip_duration:
            continue

        clips_from_video = 0
        num_attempts = 0
        while clips_from_video < args.max_clips_per_video and num_attempts < 10:
            num_attempts += 1
            clip_start = random.uniform(0, duration - args.clip_duration)

            clip_path = os.path.join(args.output_dir, f"normal_{clip_idx:04d}.mp4")
            success = extract_clip(path, clip_path, clip_start, args.clip_duration)

            if success and os.path.getsize(clip_path) > 1024:
                manifest_rows.append((clip_path, 0, fname))
                clip_idx += 1
                clips_from_video += 1
                print(f"  [{clip_idx}] {fname} @ {clip_start:.1f}s")

    # 4 --- Write manifest CSV
    manifest_path = os.path.join(args.output_dir, "clips_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_path", "label", "source_video"])
        for row in manifest_rows:
            writer.writerow(row)

    # Summary
    pos_count = sum(1 for r in manifest_rows if r[1] == 1)
    neg_count = sum(1 for r in manifest_rows if r[1] == 0)
    print(f"\n{'='*50}")
    print(f"  Done! Created {len(manifest_rows)} clips:")
    print(f"    Positive: {pos_count}")
    print(f"    Negative: {neg_count}")
    print(f"    Manifest: {manifest_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
