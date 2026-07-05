# VigilantVision — STATE.md

## Project Reference
[PROJECT.md](./PROJECT.md) — Core value: demonstrate working end-to-end concealment detection pipeline with one genuinely trained classifier.

## Current Position
- **Phase:** 1 (Environment & Data) — mostly complete, awaiting environment install
- **Plan:** None active
- **Progress:** ~60% (data extracted ✓, annotations parsed ✓, scripts ready ✓, env pending)
- **Last Action:** Migrated existing project docs into GSD `.planning/` structure

## Performance Metrics
- **Plans Completed:** 0
- **Average Duration:** N/A
- **Trend:** Just initialized

## Accumulated Context
- Dataset: ~6GB, 195 videos (50 Shoplifting + 145 Normal)
- 21 Shoplifting clips have frame-level annotations; 29 excluded from training
- Feature set: wrist_to_torso_norm, wrist_to_item_norm, num_items_nearby
- Classifier: logistic regression / small MLP (scikit-learn)
- Target FPS: usable on RTX 4060 for live webcam demo
- Train/eval: split by video, not by frame

### Key Decisions
1. Only classifier is custom-trained (rest is pretrained) — **Good**
2. Features normalized by bbox height for scale invariance — **Good**
3. COCO holdable classes as item proxy (documented limitation) — **Good**
4. Exclude unannotated Shoplifting clips from training — **Good**

### Blockers
- Environment packages not yet installed (`ultralytics`, `opencv-python`, etc.)
- Day 2 feature extraction is the next critical path

## Session Continuity
- **Last Session:** GSD project initialization
- **Last Action:** Created `.planning/` structure with migrated docs
- **Continue Here:** Run Phase 1: install dependencies → sanity-check pipeline
