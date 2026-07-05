# VigilantVision — 3D-CNN Training Guide 🎥

## Overview

This guide walks through training the **r3d_18 video classifier** (the TA-recommended approach) on a friend's laptop with an NVIDIA GPU. The model takes short 16-frame video clips and learns to distinguish Shoplifting vs Normal behavior directly from pixels using transfer learning from Kinetics-400.

**Estimated time:** ~2-4 hours (clip prep: 30-60 min, training: 1-3 hours depending on GPU)

---

## Step 1: Get the Code

### Option A: Clone from GitHub (if friend has internet)

```bash
# Works on Windows (Git Bash / PowerShell / CMD) and Linux
git clone https://github.com/Xombi17/VigilantVision.git
cd VigilantVision
```

### Option B: Copy from your USB drive

Copy the entire `VigilantVision` folder (excluding `dataset/`, `features.csv`, `.venv/`) via USB.

### Transfer the Dataset

The dataset (~6GB) is too large for GitHub. Transfer it manually:

| What | Where it is | Size |
|---|---|---|
| `dataset/` | Your laptop: `./dataset/` | ~6GB |
| `Temporal_Anomaly_Annotation.txt` | Your laptop: `./` | ~15KB |

**On Windows (friend's machine):** Just copy the `dataset/` folder and `Temporal_Anomaly_Annotation.txt` via USB/external drive into the `VigilantVision/` directory.

**On Linux:**
```bash
# On your laptop (pack):
tar czf vigilantvision_data.tar.gz dataset/ Temporal_Anomaly_Annotation.txt
# On friend's laptop (unpack):
tar xzf vigilantvision_data.tar.gz -C /path/to/VigilantVision/
```

---

## Step 2: Install Dependencies

### Step 2a: Install ffmpeg (needed for clip preparation)

**Windows:**
```powershell
# Option 1 — Using winget (Windows 10/11 built-in package manager):
winget install ffmpeg

# Option 2 — Using Chocolatey (if installed):
choco install ffmpeg

# Option 3 — Manual download:
# 1. Go to https://ffmpeg.org/download.html
# 2. Download the Windows build (gyan.dev build recommended)
# 3. Extract the ZIP, add the 'bin' folder to your PATH
# 4. Open a NEW PowerShell/CMD and test:
ffmpeg -version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg -y
```

### Step 2b: Install Python + PyTorch + Dependencies

**Windows — Using Miniconda (RECOMMENDED — easiest GPU setup):**
```powershell
# 1. Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html
#    Download "Windows 64-bit" installer and run it

# 2. Open "Anaconda Prompt" (from Start Menu) and run:
conda create -n vigilantvision python=3.10 -y
conda activate vigilantvision

# 3. Install PyTorch with CUDA (this is the key step for Windows):
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install remaining packages:
pip install av scikit-learn joblib

# 5. Verify GPU is detected:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

**Windows — Using pip directly:**
```powershell
# Make sure Python 3.10+ is installed from python.org
# Make sure NVIDIA drivers are installed from nvidia.com

# Install PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining packages:
pip install av scikit-learn joblib
```

**Linux:**
```bash
pip install torch torchvision av scikit-learn joblib --break-system-packages
```

### You should see: `CUDA available: True, Device count: 1`

> ⚠️ If it says `False`:
> - **Windows:** Reinstall NVIDIA drivers from nvidia.com → make sure "CUDA" is selected during install
> - **Windows (conda):** Make sure you used the `pytorch-cuda` package (Step 2b above)
> - **Linux:** Run `nvidia-smi` to check drivers, then `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

---

## Step 3: Prepare the Clips (30-60 min)

This script trims the UCF-Crime videos into short ~4.5-second clips centered on the annotated theft windows.

Open a terminal (PowerShell, CMD, or bash) in the `VigilantVision/` folder and run:

For **PowerShell**:
```powershell
python scripts/prepare_clips.py `
    --dataset_dir dataset `
    --annotation_file Temporal_Anomaly_Annotation.txt `
    --output_dir clips `
    --clip_duration 4.5 `
    --max_clips_per_video 5
```

For **Command Prompt (CMD)**:
```cmd
python scripts/prepare_clips.py ^
    --dataset_dir dataset ^
    --annotation_file Temporal_Anomaly_Annotation.txt ^
    --output_dir clips ^
    --clip_duration 4.5 ^
    --max_clips_per_video 5
```

For **Single line (works everywhere)**:
```bash
python scripts/prepare_clips.py --dataset_dir dataset --annotation_file Temporal_Anomaly_Annotation.txt --output_dir clips --clip_duration 4.5 --max_clips_per_video 5
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

==================================================
  Done! Created ~150-200 clips:
    Positive: ~40-60
    Negative: ~100-140
==================================================
```

---

## Step 4: Train the 3D-CNN (1-3 hours)

Make sure you're in the `VigilantVision/` folder with the conda environment activated (if using conda):

```bash
# If using conda:
conda activate vigilantvision

Run training (PowerShell):
```powershell
python scripts/train_clip_classifier.py `
    --manifest clips/clips_manifest.csv `
    --epochs 150 `
    --batch_size 4 `
    --lr 1e-4 `
    --eval_every 10
```

Run training (Command Prompt):
```cmd
python scripts/train_clip_classifier.py ^
    --manifest clips/clips_manifest.csv ^
    --epochs 150 ^
    --batch_size 4 ^
    --lr 1e-4 ^
    --eval_every 10
```

Run training (Single line):
```bash
python scripts/train_clip_classifier.py --manifest clips/clips_manifest.csv --epochs 150 --batch_size 4 --lr 1e-4 --eval_every 10
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
python scripts/train_clip_classifier.py --manifest clips/clips_manifest.csv --batch_size 2
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
| Training very slow / CPU instead of GPU | PyTorch installed without CUDA | **Windows:** Reinstall with `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` or use conda |
| "PyAV is required" error | `av` package missing | `pip install av` |
| ffmpeg/ffprobe not found | ffmpeg not installed or not in PATH | **Windows:** Install via `winget install ffmpeg` and restart terminal |
| `python3` not found on Windows | Python installed as `python` not `python3` | Use `python` instead of `python3` everywhere |
| Validation accuracy jumps wildly | Very small validation set | Normal with ~30 clips; look at trend over 150 epochs |
| `--break-system-packages` error | Only happens on Linux | **Windows:** Don't use this flag — just `pip install <pkg>` |

### Important Note:
With only 21 annotated videos, even the 3D-CNN may not achieve great metrics. **This is expected and honest.** The TA recommended this approach because it's the right *methodology* for a capstone — even if the numbers aren't production-ready, showing that you understand transfer learning, video-level cross-validation, and class imbalance strategies demonstrates real ML competence.

---

## Quick-Start Summary (Windows, for friend)

```powershell
# 1. Install ffmpeg
winget install ffmpeg

# 2. Install Miniconda (download from docs.conda.io)
#    Then open "Anaconda Prompt"

# 3. Setup environment
conda create -n vigilantvision python=3.10 -y
conda activate vigilantvision
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install av scikit-learn joblib

# 4. Clone + copy data
git clone https://github.com/Xombi17/VigilantVision.git
cd VigilantVision
#    ^^ Copy dataset/ folder here via USB ^^

# 5. Prepare clips
python scripts/prepare_clips.py --dataset_dir dataset --annotation_file Temporal_Anomaly_Annotation.txt --output_dir clips

# 6. Train!
python scripts/train_clip_classifier.py --manifest clips/clips_manifest.csv --epochs 150
```

---

## File Structure After Setup

```
VigilantVision/
├── scripts/
│   ├── extract_features.py       # Feature extraction (already ran)
│   ├── parse_annotations.py      # Annotation checker
│   ├── prepare_clips.py          # 🔥 Clip prep script (run this first)
│   ├── train_clip_classifier.py  # 🔥 3D-CNN training script (run this second)
│   └── heuristic_analysis.py     # Heuristic threshold analysis
├── clips/
│   ├── shoplifting_0001.mp4      # Generated by prepare_clips.py
│   ├── normal_0001.mp4           # Generated by prepare_clips.py
│   └── clips_manifest.csv        # Generated by prepare_clips.py
├── models/
│   └── clip_classifier_best.pt   # Best model (generated by training)
├── docs/
│   └── 3d_cnn_training_guide.md  # This guide
├── dataset/                      # UCF-Crime videos (~6GB, copy this over via USB)
├── Temporal_Anomaly_Annotation.txt
├── requirements.txt
└── README.md
```
