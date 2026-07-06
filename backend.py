import cv2
import asyncio
import base64
import json
import threading
import time
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.camera import camera_manager
from src.alerts import get_and_clear_alert
from src.engine import SurveillanceEngine

app = FastAPI(title="VigilantVision Server")

# Mount static alert assets
app.mount("/alerts", StaticFiles(directory="alerts"), name="alerts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register FastAPI Route Controllers
from src.routes import faces, settings, cameras, stats

app.include_router(faces.router)
app.include_router(settings.router)
app.include_router(cameras.router)
app.include_router(stats.router)

# WebSocket streaming memory states
latest_frame = None
lock = threading.Lock()


def video_loop():
    global latest_frame
    engine = SurveillanceEngine()

    frame_count = 0
    no_signal_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        no_signal_frame,
        "NO SIGNAL",
        (220, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        3,
    )

    TARGET_FPS = 25
    FRAME_BUDGET = 1.0 / TARGET_FPS

    while True:
        t_start = time.time()
        try:
            with camera_manager.lock:
                current_cams = list(camera_manager.cameras.items())

            frames_payload = []
            for cam_id, cam_data in current_cams:
                cap = cam_data["cap"]
                name = cam_data["name"]

                if cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        frame = no_signal_frame.copy()
                    else:
                        frame = cv2.flip(frame, 1)  # Mirror flip horizontal
                else:
                    frame = no_signal_frame.copy()

                if cap.isOpened() and "ret" in locals() and ret:
                    frame = engine.process_frame(
                        cam_id, name, frame, cam_data, frame_count
                    )

                # Encode at 480x270 with quality=72 — good balance of speed vs clarity
                resized = cv2.resize(frame, (480, 270))
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, 72]
                _, buffer = cv2.imencode(".jpg", resized, encode_params)
                jpg_as_text = base64.b64encode(buffer).decode("utf-8")
                frames_payload.append(
                    {
                        "id": cam_id,
                        "name": name,
                        "frame": jpg_as_text,
                        "status": cam_data["status"],
                    }
                )

            frame_count += 1
            if frames_payload:
                alert_payload = get_and_clear_alert()
                with lock:
                    latest_frame = {
                        "type": "multi_frame",
                        "cameras": frames_payload,
                        "alert": alert_payload,
                        "audio": "siren" if alert_payload else None,
                        "frame_count": frame_count,
                    }

        except Exception as e:
            print(f"Surveillance Loop Error: {e}")

        # Adaptive sleep: subtract time already spent so we stay close to TARGET_FPS
        elapsed = time.time() - t_start
        sleep_time = max(0.0, FRAME_BUDGET - elapsed)
        time.sleep(sleep_time)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Dashboard UI client connected to WebSocket.")

    async def receive_loop():
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass

    rx_task = asyncio.create_task(receive_loop())
    last_sent_frame = -1

    try:
        while not rx_task.done():
            message_to_send = None
            with lock:
                if latest_frame and latest_frame.get("frame_count", -1) != last_sent_frame:
                    message_to_send = json.dumps(latest_frame)
                    last_sent_frame = latest_frame.get("frame_count", -1)

            if message_to_send:
                try:
                    await websocket.send_text(message_to_send)
                except Exception as send_err:
                    print(f"WebSocket Send Error: {send_err}")
                    break
            else:
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.03)
    except Exception as e:
        print(f"WebSocket Loop Error: {e}")
    finally:
        rx_task.cancel()
        try:
            await rx_task
        except asyncio.CancelledError:
            pass
        print("Dashboard UI client disconnected.")



@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=video_loop, daemon=True)
    t.start()
    yield


app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
