# VigilantVision 🎥👁️

**Intelligent Video-Based Real-Time Retail Theft Detection and Alert System**

A 3-day capstone prototype demonstrating real-time concealment detection using computer vision. Combines pretrained YOLOv8 (detection), YOLOv8-Pose (skeletal keypoints), ByteTrack (tracking), a custom-trained lightweight classifier, and a FastAPI dashboard.

**Team VigilantVision AI:** Varad Joshi (Lead), Nathan Dsouza, Hrishikesh Nikam, Akshadha Sapre

---

## Pipeline Overview

```
Input ──► YOLOv8 ──► ByteTrack ──► YOLOv8-Pose ──► Features ──► Classifier ──► Dashboard
(feed)     (detect)    (track IDs)   (17-point      (geometric)  (trained)     (alerts)
                                        skeleton)
```

## Project Structure

```
capstone1/
├── scripts/
│   ├── extract_features.py      # Feature extraction pipeline (CURRENTLY RUNNING)
│   └── parse_annotations.py     # Annotation coverage checker
├── docs/
│   ├── PRD.md                   # Product requirements document
│   ├── ROADMAP.md               # 3-day development roadmap
│   ├── CLAUDE.md                 # AI assistant guidance
│   ├── project.md                # Project synopsis
│   └── normal_files_to_extract.txt  # Reference list of extracted Normal clips
├── .planning/                   # GSD workflow planning directory
│   ├── PROJECT.md
│   ├── REQUIREMENTS.md
│   ├── ROADMAP.md
│   ├── STATE.md
│   └── config.json
├── dataset/                     # Extracted UCF-Crime subset (6GB, gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt --break-system-packages

# 2. Check annotation coverage
python3 scripts/parse_annotations.py

# 3. Extract features (adjust --frame_stride for speed)
python3 scripts/extract_features.py \
    --dataset_dir ./dataset \
    --annotation_file Temporal_Anomaly_Annotation.txt \
    --output_csv features.csv \
    --frame_stride 3

# 4. Train classifier (coming in Phase 2)
# 5. Launch dashboard (coming in Phase 3)
```

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10+ |
| Vision | OpenCV, Ultralytics YOLOv8 / YOLOv8-Pose |
| Tracking | ByteTrack (via Ultralytics) |
| Backend | FastAPI + WebSocket |
| Classifier | scikit-learn (logistic regression / MLP) |
| Frontend | HTML5, CSS3, JavaScript |
| Storage | SQLite |

## Key Design Decisions

- **Only the classifier is custom-trained.** Detection, pose, and tracking use pretrained models (COCO weights).
- **Features are normalized** by person bounding-box height for scale invariance.
- **COCO holdable-object classes** stand in for "retail products" (documented limitation).
- **Cart detection** is simplified to a fixed polygon (not learned).
- **Train/eval split is by video, not by frame** to prevent data leakage.

## Dataset

Uses the **UCF-Crime** dataset (Sultani et al., CVPR 2018) — Shoplifting + Normal subsets only.
- 21 Shoplifting clips have frame-level theft window annotations
- ~29 additional Shoplifting clips have weak video-level labels only (excluded from training)
- 145 Normal clips provide negative examples

> **⚠️ This is a capstone prototype, not a production system.** Dataset limitations are documented transparently.

## License

MIT — see [LICENSE](LICENSE) (if applicable)
