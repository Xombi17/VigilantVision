# CLAUDE.md

Guidance for Claude (or any future contributor/session) working in this repository. Read this before making changes.

## Project Summary

VigilantVision is a 3-day capstone prototype: a real-time retail-theft/concealment detection pipeline using pretrained YOLOv8 (detection), YOLOv8-Pose (skeletal keypoints), ByteTrack (tracking), a custom-trained lightweight classifier (the one genuinely trained component), and a FastAPI dashboard.

Full context lives in `docs/PRD.md` (requirements, scope, dataset limitations) and `docs/ROADMAP.md` (day-by-day plan and status). **Read both before proposing scope changes** — the scope was deliberately and heavily cut down from an original 8-day plan to fit 3 days, and re-expanding it (e.g. "let's also train the detector") will blow the timeline.

## Hard Constraints — Do Not Violate

1. **Do not propose training YOLO detection or pose models from scratch.** They are pretrained (COCO weights) by design. The only trained component is the lightweight classifier on engineered features.
2. **Do not propose real shopping-cart detection.** It's a fixed polygon ("cart zone"), hardcoded per camera angle, documented as a known simplification. Building a real cart detector would require custom-labeled data we don't have and don't have time to create.
3. **Never claim frame-perfect or dense ground truth from UCF-Crime.** Only 21 of our extracted Shoplifting clips have frame-level annotations (from `Temporal_Anomaly_Annotation.txt`); the rest are weak/video-level only and are excluded from classifier training. Don't write report language, code comments, or metrics claims that imply otherwise.
4. **Do not extract or touch `Training-Normal-Videos` from `UCF_Crimes.zip`.** It's multi-GB per file and not needed — `Normal_Videos_for_Event_Recognition` and a size-capped subset of `Testing_Normal_Videos_Anomaly` are sufficient and already extracted.
5. **Compute assumption:** local RTX 4060 laptop is the only environment with webcam access — live-demo code must run there. Google Colab (T4) may be used for batch feature-extraction/training throughput only, never assume Colab can access a local webcam or filesystem outside its own runtime.

## Repository Layout

```
capstone1/
├── scripts/
│   ├── extract_features.py      # Feature extraction pipeline
│   └── parse_annotations.py     # Annotation coverage checker
├── docs/
│   ├── PRD.md                   # Product requirements document
│   ├── ROADMAP.md               # 3-day development roadmap
│   ├── CLAUDE.md                # This file — AI assistant guidance
│   ├── project.md               # Project synopsis
│   └── normal_files_to_extract.txt  # Reference list of extracted Normal clips
├── .planning/                   # GSD workflow planning directory
│   ├── PROJECT.md
│   ├── REQUIREMENTS.md
│   ├── ROADMAP.md
│   ├── STATE.md
│   └── config.json
├── dataset/                     # Extracted UCF-Crime subset (~6GB, gitignored)
├── Temporal_Anomaly_Annotation.txt  # Frame-level annotations (test-set subset)
├── yolov8n.pt                   # Pretrained detection model weights
├── yolov8n-pose.pt              # Pretrained pose model weights
├── features.csv                 # Feature extraction output (generated, gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

(Dashboard/backend files will be added in Day 3 — update this layout section when they land.)

## Key Technical Decisions (see PRD.md §7 for full rationale)

- **Feature set:** `wrist_to_torso_norm`, `wrist_to_item_norm`, `num_items_nearby` — all normalized by person bounding-box height for scale invariance across camera distances. Do not add unnormalized pixel-distance features; they won't generalize across clips.
- **Item detection proxy:** COCO classes curated as "holdable objects" (see `ITEM_CLASS_NAMES` in `scripts/extract_features.py`) stand in for "retail product," since COCO has no generic product class. This is a known limitation — keep it documented, don't quietly paper over it in the report.
- **Labeling logic:** frames inside an annotated Shoplifting clip's theft window(s) → positive; all frames in Normal clips → negative; frames in unannotated Shoplifting clips → excluded from training entirely (not weakly labeled), to avoid injecting label noise into a small dataset.
- **Train/eval split must be by video, not by frame.** Frames from the same clip are highly correlated; a random frame-level split will leak and inflate metrics. Always hold out whole clips.

## Commands

```bash
# Install deps
pip install -r requirements.txt --break-system-packages

# Re-check annotation coverage against local dataset
python3 scripts/parse_annotations.py

# Run feature extraction (adjust --frame_stride for speed/resolution tradeoff)
python3 scripts/extract_features.py --dataset_dir ./dataset --annotation_file Temporal_Anomaly_Annotation.txt --output_csv features.csv --frame_stride 3
```

## Reporting/Writing Guidance

When generating report text, slides, or code comments about this project:
- Be explicit about what's pretrained vs. custom-trained. Overclaiming here is the single biggest risk to credibility in a viva.
- State the UCF-Crime weak-supervision limitation plainly wherever theft-detection accuracy is discussed.
- Frame the cart-zone polygon and COCO-item-proxy as scoped simplifications with named future work, not as finished features.
