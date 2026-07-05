# VigilantVision — 3D-CNN Training Guide 🎥

## Overview

This guide walks through training the **r3d_18 video classifier** (the TA-recommended approach) on a friend's laptop with an NVIDIA GPU. The model takes short 16-frame video clips and learns to distinguish Shoplifting vs Normal behavior directly from pixels using transfer learning from Kinetics-400.

**Estimated time:** ~2-4 hours (clip prep: 30-60 min, training: 1-3 hours depending on GPU)

---

## Step 1: Get the Code

The repo is already on GitHub. Your friend can clone it:

```bash
git clone https://github.com/Xombi17/VigilantVision.git
cd VigilantVision
```

### What Else to Copy From This Laptop

The dataset and annotation file are too large for GitHub, so you need to transfer those manually:

| What | Where it is | Size |
|---|---|---|
| `dataset/` | `./dataset/` | ~6GB (UCF-Crime extracted videos) |
| `Temporal_Anomaly_Annotation.txt` | `./` | ~15KB |

**Quick copy command (USB drive):**
```bash
# On your laptop:
tar czf vigilantvision_data.tar.gz dataset/ Temporal_Anomaly_Annotation.txt
# Transfer the .tar.gz to friend's laptop, then:
tar xzf vigilantvision_data.tar.gz -C /path/to/VigilantVision/
```

---

## Step 2: Install Dependencies (Friend's Laptop)

```bash
# Install system packages for video processing (ffmpeg is needed for clip prep)
sudo apt update
sudo apt install ffmpeg -y

# Install Python packages
pip install torch torchvision av scikit-learn --break-system-packages

# Verify GPU is detected
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

**You should see:** `CUDA available: True, Device count: 1`

> **Important:** The training script checks for `av` (PyAV) at startup and will exit with a clear error if it's missing.

If it says `False`, the GPU drivers may not be set up. Install CUDA toolkit:
```bash
# Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# Check GPU drivers
nvidia-smi
```

---

## Step 3: Prepare the Clips (30-60 min)

This script trims the UCF-Crime videos into short ~4.5-second clips centered on the annotated theft windows.

```bash
cd /path/to/vigilantvision
python3 scripts/prepare_clips.py \
    --dataset_dir ./dataset \
    --annotation_file Temporal_Anomaly_Annotation.txt \
    --output_dir clips \
    --clip_duration 4.5 \
    --max_clips_per_video 5
```

**What this does:**
- Reads the annotation file to find theft window timestamps
- Extracts video clips centered on each theft window (positive examples)
- Extracts random clips from Normal videos (negative examples)
- Saves all clips to `clips/` directory as individual .mp4 files
- Creates `clips/clips_manifest.csv` with paths and labels

**Expected output:**
```
Found ~195 local videos.
Positive videos (annotated Shoplifting): 21
Negative videos (Normal): ~145

=== Extracting Positive Clips ===
  [1] Shoplifting001_x264.mp4 @ 15.2s (theft window 450-480)
  [2] Shoplifting001_x264.mp4 @ 30.5s (theft window 900-930)
  ...

=== Extracting Negative Clips ===
  [21] Normal_Videos_003_x264.mp4 @ 45.1s
  ...

==================================================
  Done! Created ~150-200 clips:
    Positive: ~40-60
    Negative: ~100-140
==================================================
```

---

## Step 4: Train the 3D-CNN (1-3 hours)

```bash
cd /path/to/vigilantvision
python3 scripts/train_clip_classifier.py \
    --manifest clips/clips_manifest.csv \
    --epochs 150 \
    --batch_size 4 \
    --lr 1e-4 \
    --eval_every 10
```

**Flags you can tweak:**
| Flag | Default | What it does |
|---|---|---|
| `--epochs` | 150 | More epochs = longer training, slightly better results |
| `--batch_size` | 4 | Set to 2 if you get CUDA OOM (out of memory) errors |
| `--lr` | 0.0001 | Learning rate. Leave at 1e-4 unless the model isn't learning |
| `--freeze_backbone` | Enabled | Freezes most of the pretrained model. Disable only if you have ~100+ clips per class |
| `--eval_every` | 10 | How often to print validation accuracy |

**What you'll see:**
```
Using device: cuda
Train clips: 120 (from distinct videos) | Val clips: 30

Epoch 10/150 | loss=0.6942 | val_acc=0.533 | prec=0.500 | rec=0.600 | f1=0.545
Epoch 20/150 | loss=0.6815 | val_acc=0.567 | prec=0.545 | rec=0.667 | f1=0.600
Epoch 30/150 | loss=0.6721 | val_acc=0.600 | prec=0.571 | rec=0.667 | f1=0.615
...
-> new best model saved (models/clip_classifier_best.pt)
...
Training complete. Best val accuracy: 0.733
```

**If you get CUDA out of memory:**
```bash
# Try smaller batch size
python3 scripts/train_clip_classifier.py --manifest clips/clips_manifest.csv --batch_size 2
```

---

## Step 5: Evaluate the Results

After training completes:

1. **Best model saved at:** `models/clip_classifier_best.pt`
2. **Metrics to report:**
   - Best validation accuracy
   - Precision, Recall, F1
   - Confusion matrix (printed at the end of training)

3. **Test on a single clip:**
```python
import torch
from torchvision.models.video import r3d_18
from torchvision.io import read_video
import torch.nn.functional as F

# Load model
model = r3d_18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("models/clip_classifier_best.pt"))
model.eval()

# Load a clip
video, _, _ = read_video("clips/normal_001.mp4", pts_unit="sec")
# ... (preprocess same as training script)
```

---

## Expected Results & Troubleshooting

### Target Metrics (what to aim for):

| Metric | Baseline (random) | Good | Great |
|---|---|---|---|
| Accuracy | 0.50 | 0.65+ | 0.75+ |
| Precision | 0.50 | 0.60+ | 0.70+ |
| Recall | 0.50 | 0.60+ | 0.70+ |
| F1 | 0.50 | 0.60+ | 0.70+ |

### Common Issues:

| Problem | Likely cause | Fix |
|---|---|---|
| Accuracy stuck at ~0.50 | Not actually learning | Check GPU is detected, lower learning rate |
| CUDA OOM | Too many clips in batch | Reduce `--batch_size` to 2 |
| Training very slow | CPU instead of GPU | `torch.cuda.is_available()` returns False → fix CUDA drivers |
| Clips fail to extract | ffmpeg not installed | `sudo apt install ffmpeg` |
| Validation accuracy jumps wildly | Very small validation set | Normal with ~30 clips; look at trend over 150 epochs |

### Important Note:
With only 21 annotated videos, even the 3D-CNN may not achieve great metrics. **This is expected and honest.** The TA recommended this approach because it's the right *methodology* for a capstone — even if the numbers aren't production-ready, showing that you understand transfer learning, video-level cross-validation, and class imbalance strategies demonstrates real ML competence.

---

## File Structure After Setup

```
vigilantvision/
├── scripts/
│   ├── extract_features.py       # Feature extraction (already ran)
│   ├── parse_annotations.py      # Annotation checker
│   ├── prepare_clips.py          # 🔥 Clip prep script (run this first)
│   └── train_clip_classifier.py  # 🔥 3D-CNN training script (run this second)
├── clips/
│   ├── shoplifting_0001.mp4      # Generated by prepare_clips.py
│   ├── normal_0001.mp4           # Generated by prepare_clips.py
│   └── clips_manifest.csv        # Generated by prepare_clips.py
├── models/
│   └── clip_classifier_best.pt   # Best model (generated by training)
├── dataset/                      # UCF-Crime videos (~6GB, copy this over)
├── Temporal_Anomaly_Annotation.txt
├── requirements.txt
└── README.md
```
