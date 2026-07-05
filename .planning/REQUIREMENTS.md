# VigilantVision — REQUIREMENTS.md

## v1 Requirements (Current Scope)

- [ ] **FR-01:** Ingest video from a file or webcam via OpenCV
- [ ] **FR-02:** Detect and track people frame-to-frame using pretrained YOLOv8 + ByteTrack
- [ ] **FR-03:** Extract 17-point pose keypoints per tracked person using pretrained YOLOv8-Pose
- [ ] **FR-04:** Detect candidate "item" objects per frame using pretrained YOLOv8 (COCO holdable-object classes as proxy)
- [ ] **FR-05:** Compute per-frame geometric features per tracked person (wrist-to-torso distance, wrist-to-item distance, nearby-item count), normalized by person bounding-box height
- [ ] **FR-06:** Train a lightweight classifier (logistic regression / small MLP) on features extracted from labeled UCF-Crime clips
- [ ] **FR-07:** Apply a fixed "cart zone" polygon as a simplification for cart-boundary logic in the live demo
- [ ] **FR-08:** Serve a FastAPI + WebSocket dashboard showing live annotated video, real-time alert popups, and a persistent SQLite incident log

## v2 Requirements (Deferred)

- [ ] **FR-09:** Export models to ONNX / quantize for performance
- [ ] **FR-10:** Package as a Docker container
- [ ] Velocity-based features (wrist speed over sliding window)

## Out of Scope

| Feature | Rationale |
|---|---|
| Training YOLO detection/pose from scratch | Using pretrained COCO weights by design; no time/data for custom training |
| Real shopping-cart detection | Requires custom-labeled data not available |
| Frame-perfect dense ground truth | UCF-Crime provides only weak labels; only 21 clips have temporal windows |
| Multi-camera RTSP ingestion | Complexity beyond 3-day scope |
| Production-hardened deployment | Capstone prototype, not production system |
| Detailed report writing | Will be done as final step |

## Traceability

| Requirement | Roadmap Phase | Status |
|---|---|---|
| FR-01 | Phase 1 (Environment) | Pending |
| FR-02 | Phase 2 (Feature Extraction) | Pending |
| FR-03 | Phase 2 (Feature Extraction) | Pending |
| FR-04 | Phase 2 (Feature Extraction) | Pending |
| FR-05 | Phase 2 (Feature Extraction) | Pending |
| FR-06 | Phase 2 (Training) | Pending |
| FR-07 | Phase 3 (Dashboard) | Pending |
| FR-08 | Phase 3 (Dashboard) | Pending |
| FR-09 | Phase 3 (Stretch) | Pending |
| FR-10 | Phase 3 (Stretch) | Pending |

### Coverage Summary
- Phase 1: FR-01
- Phase 2: FR-02, FR-03, FR-04, FR-05, FR-06
- Phase 3: FR-07, FR-08, FR-09 (stretch), FR-10 (stretch)
