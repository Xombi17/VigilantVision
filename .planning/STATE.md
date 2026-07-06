# VigilantVision — STATE.md

## Project Reference
[PROJECT.md](./PROJECT.md) — Core value: demonstrate working end-to-end concealment detection pipeline with one genuinely trained classifier.

## Current Position
- **Phase:** 3 (Dashboard, Integration & Demo Prep)
- **Plan:** Active (Implementing Cart-Zone Polygon & Sync)
- **Progress:** ~90% (data/feature extraction complete ✓, tabular classifier trained ✓, dashboard backend/frontend running ✓, cart zone in progress 🚧)
- **Last Action:** Synchronized planning documents and started cart zone implementation.

## Performance Metrics
- **Plans Completed:** 2
- **Average Duration:** N/A
- **Trend:** On track to complete today

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
- None (environment verified and functional, model trained)

## Session Continuity
- **Last Session:** GSD progress sync and roadmap update
- **Last Action:** Sync roadmap and update State
- **Continue Here:** Implement FR-07 (cart zone polygon logic) in scripts/dashboard.py and scripts/templates/index.html.
