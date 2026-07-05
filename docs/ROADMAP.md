# VigilantVision — 3-Day Roadmap

Compressed from the original 8-day plan. Scope was deliberately cut (see PRD §3, Non-Goals) to fit a 3-day capstone window while still producing a defensible, working prototype with a genuine trained component.

Compute: local RTX 4060 laptop (webcam-dependent tasks) + Google Colab T4 (batch feature extraction, if local is a bottleneck).

---

## Day 1 — Data + Environment

**Status: mostly complete**

- [x] Identify dataset structure inside `UCF_Crimes.zip` (avoid the multi-GB `Training-Normal-Videos` folder — not needed)
- [x] Extract `Anomaly-Videos/Shoplifting/*` (~50 clips, ~3-4GB)
- [x] Extract `Normal_Videos_for_Event_Recognition/*` in full (~44 clips)
- [x] Extract a size-capped subset of `Testing_Normal_Videos_Anomaly/*` (clips under 100MB, ~144 files)
- [x] Final dataset: ~6GB, ~195 local videos
- [x] Retrieve `Temporal_Anomaly_Annotation.txt` from the official CVPR2018 repo
- [x] Parse annotations against local dataset: **21 Shoplifting clips with frame-level theft windows**, **29 Shoplifting clips with weak/video-level label only**, **145 Normal clips**
- [ ] Install environment: `ultralytics`, `opencv-python`, `fastapi`, `uvicorn`, `scikit-learn`
- [ ] Sanity-check pipeline: run pretrained YOLOv8n + YOLOv8-Pose + ByteTrack on one sample video end-to-end

## Day 2 — Feature Extraction + Classifier Training

**Status: scripts ready, extraction pending**

- [ ] Run `extract_features.py` across all annotated Shoplifting + Normal clips
  - Produces `features.csv`: per-frame, per-tracked-person features (`wrist_to_torso_norm`, `wrist_to_item_norm`, `num_items_nearby`) + label
  - Tune `--frame_stride` for speed/resolution tradeoff if extraction is too slow
- [ ] Train a lightweight classifier (logistic regression or small MLP, scikit-learn/PyTorch) on `features.csv`
- [ ] Evaluate with a held-out split (e.g. leave-some-videos-out, not random-frame split, to avoid leakage between frames of the same clip)
- [ ] Report precision/recall/ROC-AUC honestly — this is a proof-of-concept, not a benchmark result
- [ ] Use the same labeled data to sanity-check/tune the geometric heuristic thresholds (fallback/baseline logic, easy to explain in a viva even if the classifier underperforms)
- [ ] (Stretch, only if ahead of schedule) Try adding velocity-based features (wrist speed over a short sliding window)

## Day 3 — Dashboard, Integration, Demo Prep

**Status: not started**

- [ ] Build FastAPI + WebSocket backend: live annotated frame stream, alert event push, SQLite incident logging
- [ ] Build a minimal HTML/CSS/JS frontend: live video panel, alert popup, incident log table
- [ ] Wire the full pipeline together: ingest → detect/track/pose → feature extraction (live, per-frame) → classifier inference → alert trigger → dashboard push
- [ ] Define and hardcode a fixed "cart zone" polygon per demo camera angle (documented simplification, not learned)
- [ ] Test on live webcam locally (RTX 4060) — expect some unpredictability, so also prepare 2-3 recorded clips (mix of annotated Shoplifting, unannotated Shoplifting, and Normal) as a reliable fallback demo
- [ ] Write up the report: architecture diagram, methodology, dataset limitations (PRD §7), metrics, and future work
- [ ] (Stretch, only if time remains) ONNX export + FP16 quantization, benchmark FPS
- [ ] (Stretch, only if time remains) Docker packaging

---

## Explicit Scope Cuts (vs. original 8-day synopsis)

| Original plan item | Status in 3-day plan |
|---|---|
| Train YOLO detector | Cut — using pretrained COCO weights |
| Train YOLO-Pose | Cut — using pretrained weights |
| Custom shopping-cart detector | Cut — replaced with fixed cart-zone polygon |
| ByteTrack integration | Kept — pretrained, no training needed |
| Geometric state machine | Kept, and backed by a genuinely trained classifier on top |
| FastAPI dashboard | Kept, full scope |
| ONNX/TensorRT optimization | Demoted to stretch goal |
| Docker packaging | Demoted to stretch goal |

This keeps exactly one real "trained model" in the pipeline (the concealment classifier), which is both honest about effort and defensible in a viva, rather than overclaiming custom-trained detection/pose models that were never realistically trainable in 3 days.
