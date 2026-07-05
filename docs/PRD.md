# VigilantVision — Product Requirements Document

**Team:** VigilantVision AI (Varad Joshi — Lead, Nathan Dsouza, Hrishikesh Nikam, Akshadha Sapre)
**Type:** Capstone project prototype
**Timeline:** 3 days (compressed from an original 8-day plan)
**Status:** Active development

---

## 1. Problem Statement

Retail shoplifting causes significant inventory loss, and traditional CCTV systems are purely passive — footage is only reviewed *after* an incident, when the loss has already occurred and the suspect is gone. VigilantVision aims to demonstrate that a lightweight, real-time computer vision pipeline can flag *suspicious concealment behavior as it happens*, giving staff a chance to intervene before a theft is completed.

This is a **capstone prototype**, not a production security product. The goal is to demonstrate a working, defensible end-to-end pipeline — not to ship a deployable retail-loss-prevention system.

## 2. Goals

- Demonstrate real-time person detection, tracking, and pose estimation on live or recorded video.
- Demonstrate a trained decision layer (classifier) that distinguishes concealment-like behavior from normal behavior, using engineered geometric features.
- Present detections through a live web dashboard with alerting and an incident log.
- Be honest and precise in the final report about what was trained vs. what was pretrained, and about the dataset's limitations (see §7).

## 3. Non-Goals (explicitly out of scope)

- **Not** a production-ready retail deployment.
- **Not** training detection/pose models from scratch — these are pretrained (COCO-based YOLOv8).
- **Not** real bounding-box-level shopping cart detection — replaced with a fixed "cart zone" polygon as a documented simplification.
- **Not** claiming frame-perfect theft detection — UCF-Crime provides only video- and (for a subset) segment-level weak supervision, not dense frame-accurate ground truth for gestures.
- **Not** multi-camera, edge-deployment, or production-hardened — these are explicitly listed as future work.

## 4. Users / Stakeholders

- **Primary:** Capstone evaluators/faculty assessing technical depth, feasibility reasoning, and honesty about limitations.
- **Secondary (illustrative persona):** A retail store security operator who would view the live dashboard and receive alerts.

## 5. Functional Requirements

| ID | Requirement | Priority |
|----|---|---|
| FR1 | Ingest video from a file or webcam via OpenCV | Must |
| FR2 | Detect and track people frame-to-frame using pretrained YOLOv8 + ByteTrack | Must |
| FR3 | Extract 17-point pose keypoints per tracked person using pretrained YOLOv8-Pose | Must |
| FR4 | Detect candidate "item" objects per frame using pretrained YOLOv8 (COCO holdable-object classes as proxy) | Must |
| FR5 | Compute per-frame geometric features per tracked person (wrist-to-torso distance, wrist-to-item distance, nearby-item count), normalized by person bounding-box height | Must |
| FR6 | Train a lightweight classifier (logistic regression / small MLP) on features extracted from labeled UCF-Crime clips | Must |
| FR7 | Apply a fixed "cart zone" polygon as a simplification for cart-boundary logic in the live demo | Should |
| FR8 | Serve a FastAPI + WebSocket dashboard showing live annotated video, real-time alert popups, and a persistent SQLite incident log | Must |
| FR9 | Export models to ONNX / quantize for performance | Could (stretch) |
| FR10 | Package as a Docker container | Could (stretch) |

## 6. Non-Functional Requirements

- Must run end-to-end on a single consumer laptop GPU (RTX 4060) at usable frame rates for a live webcam demo.
- Feature-extraction/training steps may use Google Colab (free/T4 tier) for throughput; live webcam demo must run locally, since Colab cannot access a local webcam.
- Codebase should be modular enough that any single stretch component (ONNX export, Docker) can be dropped without breaking the core demo.

## 7. Dataset & Ground-Truth Limitations (must be stated in the final report)

We use the **UCF-Crime** dataset (Sultani et al., CVPR 2018), specifically the `Shoplifting` and `Normal` subsets.

- UCF-Crime provides **video-level weak labels** for all anomaly videos, and **frame-level (temporal) annotations only for a subset of test videos** — 21 of our extracted Shoplifting clips have exact theft start/end frame ranges; the remaining ~29 extracted Shoplifting clips have no frame-level ground truth and are excluded from classifier training (they're retained only for qualitative demo playback).
- There are **no bounding-box, pose, hand, item, or cart annotations** anywhere in UCF-Crime. All spatial reasoning (pose, item boxes, cart zone) comes from pretrained models or hardcoded simplifications, not from dataset ground truth.
- The classifier is therefore trained via **weak supervision**: frames inside an annotated theft window are labeled positive, all frames in Normal clips are labeled negative, and features are engineered (not learned end-to-end) from pretrained-model outputs.
- Because of this, we do **not** claim state-of-the-art or even necessarily robust real-world theft detection — the deliverable is a **working, defensible proof-of-concept pipeline**, explicitly scoped and reported as such.

## 8. Success Criteria (capstone-appropriate)

- End-to-end pipeline runs live on webcam and on recorded demo clips without crashing.
- Classifier shows meaningfully better-than-chance separation between concealment-window frames and normal frames (report actual precision/recall/ROC-AUC honestly, whatever they are).
- Dashboard displays live feed, triggers a visible alert on a positive detection, and logs the incident to SQLite.
- Report clearly documents architecture, what's pretrained vs. trained, dataset limitations, and future work.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Live webcam demo behaves unpredictably in front of evaluators | Always have a recorded fallback clip ready |
| Classifier overfits to the small annotated-Shoplifting set (21 clips) | Keep model simple (logistic regression/small MLP), report metrics honestly, treat as proof-of-concept |
| Time runs out before Docker/ONNX stretch goals | These are explicitly Could-priority; core pipeline (FR1–FR8) is the real deliverable |
| Feature extraction on full dataset is too slow before Day 2 ends | `--frame_stride` parameter allows trading temporal resolution for speed; Normal clip count can be trimmed further if needed |

## 10. Open Questions / Future Work (for report's "Future Work" section)

- Real shopping-cart detection (would require custom-labeled data)
- Multi-camera RTSP ingestion
- Edge deployment (ONNX/TensorRT quantization)
- Larger, purpose-built (rather than repurposed) theft-detection dataset with spatial annotations
