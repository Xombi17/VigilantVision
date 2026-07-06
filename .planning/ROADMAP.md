# VigilantVision — ROADMAP.md

A 3-day capstone prototype: real-time retail-theft detection using pretrained YOLOv8 + YOLOv8-Pose + ByteTrack + custom classifier + FastAPI dashboard. Scope deliberately cut from original 8-day plan to fit 3 days with one genuinely trained component.

---

## Phase 1 — Environment & Data

**Status:** ✅ Complete

**Goals:** Get the pipeline environment ready and sanity-check the detection/tracking stack.

**Dependencies:** None

**Requirements:** FR-01

**Success Criteria:** Pretrained YOLOv8n, YOLOv8-Pose, ByteTrack run end-to-end on one sample video

**Files:**
- `parse_annotations.py` ✓
- `extract_features.py` ✓
- Environment install (ultralytics, opencv, fastapi, etc.) ✓
- Sanity-check pipeline on one sample video ✓

### Plans
- **01-01:** Install Python dependencies (`pip install ultralytics opencv-python fastapi uvicorn scikit-learn`) ✓
- **01-02:** Run YOLOv8n + YOLOv8-Pose + ByteTrack on one sample video to verify end-to-end detection/tracking/pose works ✓

---

## Phase 2 — Feature Extraction & Classifier Training

**Status:** ✅ Complete

**Goals:** Extract geometric features from all annotated clips, train + evaluate the lightweight classifier.

**Dependencies:** Phase 1 (environment ready)

**Requirements:** FR-02, FR-03, FR-04, FR-05, FR-06

**Success Criteria:** Classifier shows meaningfully better-than-chance separation (report precision/recall/ROC-AUC honestly)

**Files:**
- `extract_features.py` ✓
- `features.csv` (generated) ✓
- Classifier training script (`train_tabular_classifier.py`) ✓

### Plans
- **02-01:** Run `extract_features.py` across all annotated Shoplifting + Normal clips (tune `--frame_stride`) ✓
- **02-02:** Train logistic regression / small MLP on `features.csv` with leave-some-videos-out split ✓
- **02-03:** Evaluate and report metrics honestly; tune heuristic thresholds as fallback ✓
- **02-04:** (Stretch) Add velocity-based features over sliding window

---

## Phase 3 — Dashboard, Integration & Demo

**Status:** 🚧 In Progress

**Goals:** Build the FastAPI dashboard with live video, alerts, and incident logging. Wire the full pipeline together.

**Dependencies:** Phase 2 (trained classifier)

**Requirements:** FR-07, FR-08, FR-09 (stretch), FR-10 (stretch)

**Success Criteria:** End-to-end pipeline runs on webcam without crashing; dashboard displays feed, triggers alerts, logs incidents

**Files:**
- Dashboard backend (FastAPI + WebSocket)
- Dashboard frontend (HTML/CSS/JS)
- Incident log (SQLite)
- Cart-zone polygon configuration

### Plans
- **03-01:** Build FastAPI + WebSocket backend (live frame stream, alert push, SQLite logging)
- **03-02:** Build minimal HTML/CSS/JS frontend (live video, alert popup, incident log)
- **03-03:** Wire full pipeline: ingest → detect/track/pose → feature extraction → classifier → alert → dashboard
- **03-04:** Define cart-zone polygon per camera angle
- **03-05:** Test on live webcam + prepare recorded fallback clips
- **03-06:** Write up report (architecture, methodology, limitations, metrics, future work)
- **03-07:** (Stretch) ONNX export + FP16 quantization, benchmark FPS
- **03-08:** (Stretch) Docker packaging

---

## Governance

- Phase numbering allows inserted work (e.g., Phase 2.1) if urgent changes arise between primary phases
- Each plan file follows the naming convention: `{phase}-{plan}-PLAN.md`
- Success criteria are defined as observable behaviors, not time estimates
- Completed phases are collapsed into `<details>` tags with ✅ status indicators
