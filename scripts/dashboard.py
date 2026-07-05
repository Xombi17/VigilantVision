"""
VigilantVision — Real-Time Detection Dashboard
================================================
FastAPI + WebSocket dashboard that:
  - Streams live annotated video (webcam or recorded clip)
  - Runs YOLOv8-Pose + YOLOv8 detection on each frame
  - Computes geometric features (wrist_to_torso, wrist_to_item, item_count)
  - Runs the trained tabular classifier to score each tracked person
  - Fires visual alerts when concealment confidence exceeds threshold
  - Logs incidents to SQLite for review / viva demo

Usage:
    # Webcam live demo
    python scripts/dashboard.py --source webcam

    # Pre-recorded clip
    python scripts/dashboard.py --source dataset/Shoplifting001_x264.mp4

    # Options
    python scripts/dashboard.py --source webcam --threshold 0.3 --port 8000
"""

import os
import sys
import json
import argparse
import asyncio
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import joblib
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent.resolve()
ROOT_DIR  = APP_DIR.parent
TEMPLATES_DIR = APP_DIR / "templates"
DB_PATH  = ROOT_DIR / "incidents.db"

# COCO 17-point skeleton indices
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP           = 11, 12
L_WRIST, R_WRIST       = 9, 10
L_ELBOW, R_ELBOW       = 7, 8

# COCO classes considered "holdable" (stand-in for retail products)
ITEM_CLASS_NAMES = {
    "backpack", "umbrella", "handbag", "suitcase", "bottle", "wine glass",
    "cup", "banana", "apple", "sandwich", "orange", "book", "vase",
    "scissors", "teddy bear", "toothbrush", "cell phone", "remote",
    "keyboard", "mouse", "laptop", "bowl", "clock",
}

SENTINEL = 1_000_000.0


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            video_source TEXT,
            frame_idx   INTEGER,
            track_id    INTEGER,
            confidence  REAL,
            features    TEXT,
            dismissed   INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


def log_incident(video_source, frame_idx, track_id, confidence, feat_dict):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT INTO incidents (timestamp, video_source, frame_idx, track_id, confidence, features) "
        "VALUES (?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(),
         str(video_source), frame_idx, track_id,
         float(confidence), json.dumps(feat_dict)),
    )
    conn.commit()
    conn.close()


def get_incidents(limit=100):
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------
def _dist(p1, p2):
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _kp_valid(kp):
    """Return True if a keypoint was detected (not at origin)."""
    return not np.all(kp == 0)


def draw_skeleton(frame, kpts, color=(0, 255, 0), thickness=2):
    ls, rs = kpts[L_SHOULDER], kpts[R_SHOULDER]
    lh, rh = kpts[L_HIP], kpts[R_HIP]
    lw, rw = kpts[L_WRIST], kpts[R_WRIST]
    le, re = kpts[L_ELBOW], kpts[R_ELBOW]

    # Torso
    if _kp_valid(ls) and _kp_valid(rs):
        cv2.line(frame, (int(ls[0]), int(ls[1])), (int(rs[0]), int(rs[1])), color, thickness-1)
    for sh, hp in [(ls, lh), (rs, rh)]:
        if _kp_valid(sh) and _kp_valid(hp):
            cv2.line(frame, (int(sh[0]), int(sh[1])), (int(hp[0]), int(hp[1])), color, thickness-1)

    # Arms
    for wr, el in [(lw, le), (rw, re)]:
        if _kp_valid(wr) and _kp_valid(el):
            cv2.line(frame, (int(wr[0]), int(wr[1])), (int(el[0]), int(el[1])), color, thickness)
    for el, sh in [(le, ls), (re, rs)]:
        if _kp_valid(el) and _kp_valid(sh):
            cv2.line(frame, (int(el[0]), int(el[1])), (int(sh[0]), int(sh[1])), color, thickness)

    # Joint dots
    for j in kpts:
        if _kp_valid(j):
            cv2.circle(frame, (int(j[0]), int(j[1])), 4, (0, 255, 255), -1)


# ---------------------------------------------------------------------------
# App factory  (called from main())
# ---------------------------------------------------------------------------
def create_app(video_source="webcam", model_path=None,
               threshold=0.3, frame_interval=2):
    # --- resolve source ---------------------------------------------------
    if isinstance(video_source, str) and video_source in ("webcam", "0"):
        cap_source = 0
        source_label = "Webcam"
    elif isinstance(video_source, str) and video_source.isdigit():
        cap_source = int(video_source)
        source_label = f"Camera {cap_source}"
    else:
        cap_source = video_source
        source_label = Path(str(video_source)).name

    # --- load models ------------------------------------------------------
    print("Loading YOLO models …")
    pose_model = YOLO(str(ROOT_DIR / "yolov8n-pose.pt"))
    item_model = YOLO(str(ROOT_DIR / "yolov8n.pt"))

    # --- load classifier --------------------------------------------------
    classifier = None
    scaler = None
    if model_path and os.path.exists(model_path):
        print(f"Loading classifier from {model_path} …")
        try:
            cp = joblib.load(model_path)
            classifier = cp["model"]
            scaler     = cp["scaler"]
            print(f"  ✓ {type(classifier).__name__} loaded")
        except Exception as e:
            print(f"  ⚠ Could not load classifier: {e}")
    else:
        print("  Running in detection-only mode (no classifier)")

    # --- FastAPI app ------------------------------------------------------
    app = FastAPI(title="VigilantVision Dashboard")

    @app.on_event("startup")
    async def _startup():
        init_db()
        print(f"  DB: {DB_PATH}")

    # ---- WebSocket live stream --------------------------------------------
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        print("WebSocket connected")

        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            await ws.send_json({"type": "error",
                                "message": f"Cannot open source: {source_label}"})
            await ws.close()
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        await ws.send_json({
            "type": "info",
            "fps": round(fps, 1), "width": w, "height": h,
            "source": source_label,
            "threshold": threshold,
            "frame_interval": frame_interval,
        })

        frame_idx = 0
        proc_times = []
        alert_cooldown = 0
        recent_incidents = set()  # dedup: (track_id, bucket_of_50_frames)
        current_threshold = threshold  # mutable local, can be updated via WS messages
        jpeg_quality = 78  # mutable local, can be updated via WS messages

        async def recv_loop():
            """Background task that reads control messages from the client."""
            nonlocal current_threshold, jpeg_quality
            try:
                while True:
                    raw = await ws.receive_text()
                    msg = json.loads(raw)
                    cmd = msg.get("cmd", "")
                    if cmd == "set_threshold":
                        val = float(msg["value"])
                        current_threshold = max(0.0, min(1.0, val))
                        print(f"  Threshold updated → {current_threshold:.2f}")
                    elif cmd == "set_quality":
                        val = int(msg["value"])
                        jpeg_quality = max(10, min(100, val))
                        print(f"  JPEG quality updated → {jpeg_quality}")
            except (WebSocketDisconnect, Exception):
                pass

        async def stream_loop():
            """Main loop: read frames, process, send."""
            nonlocal frame_idx, proc_times, alert_cooldown, recent_incidents
            nonlocal current_threshold, jpeg_quality

            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(cap_source, int):
                        await asyncio.sleep(0.01)
                        continue
                    else:
                        await ws.send_json({"type": "video_end"})
                        await ws.close()
                        break

                frame_idx += 1
                if frame_idx % frame_interval != 0:
                    continue

                t0 = time.perf_counter()

                # ---- YOLO runs --------------------------------------------
                pose_res = pose_model.track(frame, persist=True,
                                            verbose=False, classes=[0])
                item_res = item_model.predict(frame, verbose=False)

                # Item boxes  (cx, cy, x1, y1, x2, y2)
                item_boxes = []
                if item_res and item_res[0].boxes is not None:
                    names = item_res[0].names
                    for box in item_res[0].boxes:
                        cid = int(box.cls[0])
                        if names.get(cid, "") not in ITEM_CLASS_NAMES:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cx, cy = (x1+x2)/2, (y1+y2)/2
                        item_boxes.append((cx, cy, x1, y1, x2, y2))

                # ---- Annotate frame ---------------------------------------
                annotated = frame.copy()

                # Draw item boxes (blue)
                for ib in item_boxes:
                    _, _, x1, y1, x2, y2 = ib
                    cv2.rectangle(annotated, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (255, 140, 0), 2)
                    cv2.putText(annotated, "item", (int(x1), int(y1)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 140, 0), 1)

                # ---- Per-person processing --------------------------------
                persons = []
                max_conf = 0.0
                any_alert = False
                logged_this_frame = False

                if (pose_res and pose_res[0].keypoints is not None
                        and pose_res[0].boxes is not None):
                    kpts_all = pose_res[0].keypoints.xy.cpu().numpy()
                    boxes_xyxy = pose_res[0].boxes.xyxy.cpu().numpy()
                    track_ids = (pose_res[0].boxes.id.cpu().numpy().astype(int)
                                 if pose_res[0].boxes.id is not None
                                 else np.full(len(kpts_all), -1))

                    for pi, kpts in enumerate(kpts_all):
                        ls, rs = kpts[L_SHOULDER], kpts[R_SHOULDER]
                        lh, rh = kpts[L_HIP], kpts[R_HIP]
                        lw, rw = kpts[L_WRIST], kpts[R_WRIST]

                        if np.all(ls == 0) or np.all(lh == 0):
                            continue

                        # Torso centre
                        tc_x = (ls[0] + rs[0] + lh[0] + rh[0]) / 4
                        tc_y = (ls[1] + rs[1] + lh[1] + rh[1]) / 4
                        torso_centre = (tc_x, tc_y)

                        # Person height
                        x1, y1, x2, y2 = boxes_xyxy[pi]
                        pheight = max(y2 - y1, 1.0)

                        # --- Features ------------------------------------
                        wtt = min(
                            _dist(lw, torso_centre) if _kp_valid(lw) else 1e6,
                            _dist(rw, torso_centre) if _kp_valid(rw) else 1e6,
                        ) / pheight

                        if item_boxes:
                            wti = min(
                                min(_dist(lw, (ib[0], ib[1])) for ib in item_boxes)
                                if _kp_valid(lw) else 1e6,
                                min(_dist(rw, (ib[0], ib[1])) for ib in item_boxes)
                                if _kp_valid(rw) else 1e6,
                            ) / pheight
                        else:
                            wti = 1e6

                        ni = sum(
                            1 for ib in item_boxes
                            if (_kp_valid(lw)
                                and _dist(lw, (ib[0], ib[1])) / pheight < 1.5)
                            or (_kp_valid(rw)
                                and _dist(rw, (ib[0], ib[1])) / pheight < 1.5)
                        )

                        # --- Classifier ----------------------------------
                        confidence = 0.0
                        if classifier is not None and scaler is not None:
                            feat = np.array([[wtt, wti, ni]], dtype=np.float64)
                            if abs(feat[0, 1] - SENTINEL) < 1.0:
                                feat[0, 1] = float("nan")
                            if np.isnan(feat[0, 1]):
                                feat[0, 1] = 0.0
                            feat_scaled = scaler.transform(feat)
                            confidence = float(
                                classifier.predict_proba(feat_scaled)[0, 1])

                        max_conf = max(max_conf, confidence)
                        tid = int(track_ids[pi])

                        persons.append({
                            "track_id": tid,
                            "confidence": round(confidence, 4),
                            "features": {
                                "wtt": round(wtt, 3),
                                "wti": round(wti, 3),
                                "ni":  ni,
                            },
                        })

                        is_alert = confidence >= current_threshold
                        if is_alert:
                            any_alert = True

                        # --- Drawing --------------------------------------
                        skel_color = (0, 255, 0) if not is_alert else (0, 0, 255)
                        draw_skeleton(annotated, kpts, color=skel_color)

                        box_color = (0, 255, 0) if not is_alert else (0, 0, 255)
                        cv2.rectangle(annotated, (int(x1), int(y1)),
                                      (int(x2), int(y2)), box_color, 2)
                        tid_label = f"ID:{tid} {confidence:.2f}"
                        cv2.putText(annotated, tid_label,
                                    (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    box_color, 2)

                        # --- Log incident once per ~50 frame window -------
                        if is_alert and not logged_this_frame:
                            bucket = (tid, frame_idx // 50)
                            if bucket not in recent_incidents:
                                recent_incidents.add(bucket)
                                if len(recent_incidents) > 100:
                                    recent_incidents.pop()
                                log_incident(
                                    source_label, frame_idx, tid,
                                    confidence,
                                    {"wtt": round(wtt, 3),
                                     "wti": round(wti, 3),
                                     "ni": ni},
                                )
                                logged_this_frame = True

                # ---- HUD overlay -----------------------------------------
                cv2.putText(annotated, "VigilantVision", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (220, 220, 220), 2)
                cv2.putText(annotated, f"src: {source_label}  "
                            f"fr:{frame_idx}  thr:{current_threshold:.2f}",
                            (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (180, 180, 180), 1)

                # Large alert banner (with cooldown to avoid flicker)
                if any_alert and alert_cooldown <= 0:
                    alert_cooldown = 20
                if alert_cooldown > 0:
                    alert_cooldown -= 1
                    cv2.putText(annotated, "⚠  CONCEALMENT DETECTED  ⚠",
                                (w//2 - 220, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                                (0, 0, 255), 5)
                    cv2.rectangle(annotated,
                                  (w//2 - 240, h//2 - 35),
                                  (w//2 + 240, h//2 + 35),
                                  (0, 0, 255), 3)

                # ---- Timing ----------------------------------------------
                dt_ms = (time.perf_counter() - t0) * 1000
                proc_times.append(dt_ms)
                if len(proc_times) > 30:
                    proc_times.pop(0)
                avg_ms = np.mean(proc_times)

                # ---- Encode & send ---------------------------------------
                _, buf = cv2.imencode(".jpg", annotated,
                                      [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                await ws.send_bytes(buf.tobytes())
                await ws.send_json({
                    "type": "frame",
                    "frame_idx": frame_idx,
                    "processing_ms": round(avg_ms, 1),
                    "num_persons": len(persons),
                    "persons": persons,
                    "max_confidence": round(max_conf, 4),
                    "alert": any_alert,
                })

        # Run stream loop and recv loop concurrently
        try:
            await asyncio.gather(stream_loop(), recv_loop())
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as exc:
            print(f"Stream error: {exc}")
        finally:
            cap.release()

    # ---- REST endpoints --------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index():
        html = (TEMPLATES_DIR / "index.html").read_text()
        return html

    @app.get("/api/incidents")
    async def list_incidents(limit: int = 100):
        return get_incidents(limit)

    @app.get("/api/stats")
    async def get_stats():
        conn = sqlite3.connect(str(DB_PATH))
        total   = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        last_h  = conn.execute(
            "SELECT COUNT(*) FROM incidents WHERE timestamp > ?",
            ((datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),),
        ).fetchone()[0]
        conn.close()
        return {"total_incidents": total, "last_hour": last_h}

    return app


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VigilantVision Real-Time Dashboard")
    parser.add_argument("--source", default="webcam",
                        help="Video source: 'webcam', camera index, or .mp4 path")
    parser.add_argument("--model",
                        default=str(ROOT_DIR / "models"
                                    / "tabular_classifier_full.joblib"),
                        help="Path to trained classifier .joblib")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Alert confidence threshold (0.0-1.0)")
    parser.add_argument("--frame-interval", type=int, default=2,
                        help="Process every Nth frame")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-classifier", action="store_true",
                        help="Run detection-only mode")
    args = parser.parse_args()

    model_path = None if args.no_classifier else args.model

    # Validate model path early
    if model_path and not os.path.exists(model_path):
        print(f"⚠ Model file not found: {model_path}")
        print("  Either place the trained model, or use --no-classifier")
        sys.exit(1)

    app = create_app(
        video_source=args.source,
        model_path=model_path,
        threshold=args.threshold,
        frame_interval=args.frame_interval,
    )

    print(f"\n{'═'*60}")
    print(f"  ██╗   ██╗██╗ ██████╗ ██╗██╗      █████╗ ███╗   ██╗████████╗")
    print(f"  ██║   ██║██║██╔════╝ ██║██║     ██╔══██╗████╗  ██║╚══██╔══╝")
    print(f"  ██║   ██║██║██║  ███╗██║██║     ███████║██╔██╗ ██║   ██║   ")
    print(f"  ╚██╗ ██╔╝██║██║   ██║██║██║     ██╔══██║██║╚██╗██║   ██║   ")
    print(f"   ╚████╔╝ ██║╚██████╔╝██║███████╗██║  ██║██║ ╚████║   ██║   ")
    print(f"    ╚═══╝  ╚═╝ ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ")
    print(f"{'═'*60}")
    print(f"  Source:     {args.source}")
    print(f"  Threshold:  {args.threshold}")
    print(f"  Frame int:  {args.frame_interval}")
    print(f"  Port:       {args.port}")
    print(f"  Classifier: {'Enabled' if not args.no_classifier else 'Detection only'}")
    print(f"{'═'*60}")
    print(f"  Open → http://localhost:{args.port}")
    print(f"{'═'*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
