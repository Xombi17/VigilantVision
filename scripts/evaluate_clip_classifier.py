"""
VigilantVision - Independent Evaluation of clip_classifier_best.pt
======================================================================
Run this to get an HONEST accuracy/precision/recall number for the trained
checkpoint, independent of whatever the training run itself reported.

Re-creates the SAME video-level train/val split (seed=42) used in
train_clip_classifier.py, so this evaluates on the same held-out videos
the model was validated against during training -- if numbers here don't
match what you saw during training, something is inconsistent (different
manifest, different seed, or the checkpoint isn't actually from this split).

Usage:
    python3 evaluate_clip_classifier.py \
        --manifest clips/clips_manifest.csv \
        --checkpoint models/clip_classifier_best.pt
"""

import csv
import random
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18
from torchvision.io import read_video
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

NUM_FRAMES = 16
FRAME_SIZE = 112


def load_manifest(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["clip_path"], int(row["label"]), row["source_video"]))
    return rows


def video_level_split(rows, val_fraction=0.2):
    videos_by_label = defaultdict(set)
    for _, label, source in rows:
        videos_by_label[label].add(source)

    val_videos = set()
    for label, videos in videos_by_label.items():
        videos = list(videos)
        random.shuffle(videos)
        n_val = max(1, int(len(videos) * val_fraction))
        val_videos.update(videos[:n_val])

    train_rows = [r for r in rows if r[2] not in val_videos]
    val_rows = [r for r in rows if r[2] in val_videos]
    return train_rows, val_rows


class ClipDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        clip_path, label, source = self.rows[idx]
        video, _, _ = read_video(clip_path, pts_unit="sec")

        total_frames = video.shape[0]
        if total_frames >= NUM_FRAMES:
            indices = torch.linspace(0, total_frames - 1, NUM_FRAMES).long()
        else:
            indices = torch.cat(
                [
                    torch.arange(total_frames),
                    torch.full((NUM_FRAMES - total_frames,), total_frames - 1),
                ]
            ).long()

        frames = video[indices].float() / 255.0
        frames = frames.permute(0, 3, 1, 2)
        frames = F.interpolate(
            frames, size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False
        )

        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
        frames = (frames - mean) / std

        clip_tensor = frames.permute(1, 0, 2, 3)
        return clip_tensor, torch.tensor(label, dtype=torch.float32), clip_path, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="clips/clips_manifest.csv")
    parser.add_argument("--checkpoint", default="models/clip_classifier_best.pt")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = load_manifest(args.manifest)
    train_rows, val_rows = video_level_split(rows, val_fraction=0.2)

    train_videos = set(r[2] for r in train_rows)
    val_videos = set(r[2] for r in val_rows)
    overlap = train_videos & val_videos
    print(f"Train clips: {len(train_rows)} from {len(train_videos)} videos")
    print(f"Val clips:   {len(val_rows)} from {len(val_videos)} videos")
    print(f"Video overlap between train/val (MUST be 0): {len(overlap)}")
    if overlap:
        print(
            "  !! WARNING: leakage detected -- these videos appear in both sets:",
            overlap,
        )

    val_loader = DataLoader(
        ClipDataset(val_rows), batch_size=args.batch_size, shuffle=False
    )

    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_paths, all_sources, all_probs = [], [], [], [], []

    with torch.no_grad():
        for clips, labels, paths, sources in val_loader:
            clips = clips.to(device)
            logits = model(clips).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
            all_paths.extend(paths)
            all_sources.extend(sources)

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
    )

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n=== HELD-OUT VALIDATION METRICS (video-level split) ===")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"ROC-AUC:   {auc:.3f}")
    except ValueError:
        print("ROC-AUC: undefined (only one class present in val set)")
    print(f"\nConfusion matrix [[TN, FP], [FN, TP]]:\n{cm}")

    print("\n=== Per-clip results (val set) ===")
    for path, source, label, pred, prob in zip(
        all_paths, all_sources, all_labels, all_preds, all_probs
    ):
        flag = "  <-- WRONG" if label != pred else ""
        print(
            f"  {path} (from {source}) | true={int(label)} pred={int(pred)} prob={prob:.3f}{flag}"
        )


if __name__ == "__main__":
    main()
