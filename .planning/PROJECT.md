# VigilantVision — PROJECT.md

## What This Is

VigilantVision is a 3-day capstone prototype that detects retail theft/concealment behavior in real time using computer vision. It combines pretrained YOLOv8 (detection), YOLOv8-Pose (skeletal keypoints), ByteTrack (tracking), a custom-trained lightweight classifier (the only genuinely trained component), and a FastAPI dashboard. Built by Team VigilantVision AI for a capstone evaluation audience (faculty assessing technical depth and honest scoping).

**Status:** Day 1 mostly complete (data extracted, annotations parsed). Day 2–3 pending.

## Core Value

Demonstrate a working, defensible end-to-end pipeline for real-time concealment detection using a single custom-trained classifier on top of pretrained vision models — not ship a production security product.

## Requirements

### Validated
- UCF-Crime Shoplifting + Normal subset extracted (~6GB, ~195 videos)
- `Temporal_Anomaly_Annotation.txt` parsed: 21 clips with frame-level windows, 29 clip-level only, 145 Normal
- Feature extraction script (`extract_features.py`) ready
- Pretrained weights (YOLOv8n, YOLOv8n-Pose) available locally

### Active
- **FR1:** Ingest video from file or webcam via OpenCV
- **FR2:** Detect & track people frame-to-frame with YOLOv8 + ByteTrack
- **FR3:** Extract 17-point pose keypoints per tracked person via YOLOv8-Pose
- **FR4:** Detect candidate "item" objects per frame (COCO holdable-class proxy)
- **FR5:** Compute per-frame geometric features (wrist-to-torso, wrist-to-item, item count) normalized by bbox height
- **FR6:** Train lightweight classifier (logistic regression / small MLP) on extracted features
- **FR7:** Fixed "cart zone" polygon for live demo
- **FR8:** FastAPI + WebSocket dashboard with live video, alerts, SQLite incident log

### Out of Scope
| Feature | Rationale |
|---|---|
| Training YOLO detection/pose from scratch | Using pretrained COCO weights by design |
| Real shopping-cart detection | Replaced with fixed cart-zone polygon |
| Frame-perfect theft ground truth | Only 21 clips have temporal annotations |
| Multi-camera RTSP | Explicit future work |
| Edge deployment / ONNX | Stretch goal only |
| Docker packaging | Stretch goal only |

## Context

- **Timeline:** 3 days, compressed from original 8-day plan
- **Compute:** Local RTX 4060 laptop (webcam) + Google Colab T4 (batch extraction)
- **Dataset:** UCF-Crime (Sultani et al., CVPR 2018) — Shoplifting + Normal subsets only
- **Weak supervision:** Only 21 of ~50 Shoplifting clips have frame-level labels; remaining 29 excluded from training
- **Item proxy:** COCO holdable-object classes stand in for "retail product" (known limitation)
- **Predecessor docs:** `PRD.md`, `ROADMAP.md`, `project.md`, `CLAUDE.md` (migrated into this structure)

## Constraints

1. **Do not train YOLO detection or pose models** — they are pretrained COCO weights.
2. **Do not implement real cart detection** — use fixed polygon simplification.
3. **Never claim frame-perfect ground truth** — only 21 clips have temporal annotations.
4. **Do not extract `Training-Normal-Videos`** — multi-GB, not needed.
5. **Webcam demo must run locally** — Colab cannot access local webcam/filesystem.
6. **Train/eval split must be by video, not by frame** — prevents data leakage.

## Key Decisions

| Decision | Rationale | Status |
|---|---|---|
| Only classifier is custom-trained | Keeps scope honest and defensible in viva | Good |
| Features: wrist-to-torso, wrist-to-item, item count | All normalized by bbox height for scale invariance | Good |
| COCO holdable classes as item proxy | No generic product class in COCO | Good (documented limitation) |
| Exclude unannotated Shoplifting clips from training | Avoids injecting label noise | Good |
| Frame-stride parameter for speed tuning | Allows trading temporal resolution for throughput | Good |
| Logistic regression / small MLP classifier | Simple enough to avoid overfitting on 21 clips | Pending validation |

## Evolution

- Updates happen at phase transitions and milestone reviews
- STATE.md is the authoritative source for current session position
- See `.planning/REQUIREMENTS.md` for requirement traceability
- See `.planning/ROADMAP.md` for phase structure
