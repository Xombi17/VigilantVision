"""
VigilantVision — 3D-CNN Video Clip Classifier Training
=======================================================
Fine-tunes a Kinetics-pretrained r3d_18 (3D-CNN) on short Shoplifting vs
Normal clips, using the TA-recommended strategy:

  - Small batch size (2-4) — video clips are memory-heavy.
  - Stratified sampling per epoch: all positive clips + random subset of
    negatives, to handle class imbalance.
  - Many epochs (150-200) so random resampling accumulates coverage.
  - Video-level train/val split (never split clips from same source video).

Usage:
    pip install torch torchvision av scikit-learn --break-system-packages
    python3 scripts/train_clip_classifier.py
        --manifest clips/clips_manifest.csv
        --epochs 150

Requirements:
    - Torch, torchvision, av (for video loading)
    - NVIDIA GPU with >=8GB VRAM recommended
"""

import os
import sys
import csv
import random
import argparse
from collections import defaultdict

# Check that PyAV is installed (torchvision.io.read_video needs it)
try:
    import av  # noqa: F401
except ImportError:
    sys.exit(
        "PyAV is required for video loading. Install it with:\n"
        "  pip install av --break-system-packages"
    )

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision.io import read_video
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)

NUM_FRAMES = 16  # frames sampled per clip, matches typical Kinetics protocol
FRAME_SIZE = 112  # r3d_18 expects 112x112 input


def load_manifest(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["clip_path"], int(row["label"]), row["source_video"]))
    return rows


def video_level_split(rows, val_fraction=0.2):
    """Split by unique source_video, stratified by label, to avoid leakage."""
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
        clip_path, label, _ = self.rows[idx]
        video, _, _ = read_video(clip_path, pts_unit="sec")  # (T, H, W, C), uint8

        total_frames = video.shape[0]
        if total_frames >= NUM_FRAMES:
            indices = torch.linspace(0, total_frames - 1, NUM_FRAMES).long()
        else:
            # pad by repeating the last frame if the clip is short
            indices = torch.cat(
                [
                    torch.arange(total_frames),
                    torch.full((NUM_FRAMES - total_frames,), total_frames - 1),
                ]
            ).long()

        frames = video[indices].float() / 255.0  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        frames = F.interpolate(
            frames, size=(FRAME_SIZE, FRAME_SIZE), mode="bilinear", align_corners=False
        )

        # Kinetics-400 normalization stats
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
        frames = (frames - mean) / std

        clip_tensor = frames.permute(
            1, 0, 2, 3
        )  # (C, T, H, W) — model's expected layout
        return clip_tensor, torch.tensor(label, dtype=torch.float32)


def stratified_epoch_sample(train_rows, max_per_class=50):
    pos = [r for r in train_rows if r[1] == 1]
    neg = [r for r in train_rows if r[1] == 0]

    pos_sample = pos if len(pos) <= max_per_class else random.sample(pos, max_per_class)
    neg_sample = neg if len(neg) <= max_per_class else random.sample(neg, max_per_class)

    epoch_rows = pos_sample + neg_sample
    random.shuffle(epoch_rows)
    return epoch_rows


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    acc = correct / max(total, 1)
    return acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="clips/clips_manifest.csv")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_per_class_per_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--checkpoint", default="models/clip_classifier_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rows = load_manifest(args.manifest)
    train_rows, val_rows = video_level_split(rows, val_fraction=0.2)
    print(
        f"Train clips: {len(train_rows)} (from distinct videos) | Val clips: {len(val_rows)}"
    )

    val_loader = DataLoader(
        ClipDataset(val_rows), batch_size=args.batch_size, shuffle=False
    )

    # --- Model: transfer learning from Kinetics-400 ---
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)

    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the last residual block + final classifier for fine-tuning
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)  # binary output (logit)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rows = stratified_epoch_sample(train_rows, args.max_per_class_per_epoch)
        train_loader = DataLoader(
            ClipDataset(epoch_rows), batch_size=args.batch_size, shuffle=True
        )

        total_loss = 0.0
        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(clips).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * clips.size(0)

        avg_loss = total_loss / max(len(epoch_rows), 1)

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_acc, all_preds, all_labels = evaluate(model, val_loader, device)

            # Calculate additional metrics
            from sklearn.metrics import precision_score, recall_score, f1_score

            try:
                precision = precision_score(all_labels, all_preds, zero_division=0)
                recall = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)
                print(
                    f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | "
                    f"val_acc={val_acc:.3f} | prec={precision:.3f} | "
                    f"rec={recall:.3f} | f1={f1:.3f}"
                )
            except Exception as e:
                print(
                    f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.3f} | "
                    f"sklearn metrics unavailable: {e}"
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), args.checkpoint)
                print(f"  -> new best model saved ({args.checkpoint})")
        else:
            print(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Best model checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
