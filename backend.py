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
    no_signal_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.putText(
        no_signal_frame,
        "NO SIGNAL",
        (400, 360),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        3,
    )

    while True:
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

                # Resize to standard and encode for streaming
                resized = cv2.resize(frame, (640, 360))
                _, buffer = cv2.imencode(".jpg", resized)
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
                    }

            time.sleep(0.04)

        except Exception as e:
            print(f"Surveillance Loop Error: {e}")
            time.sleep(1)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Dashboard UI client connected to WebSocket.")
    try:
        while True:
            message_to_send = None
            with lock:
                if latest_frame:
                    message_to_send = json.dumps(latest_frame)

            if message_to_send:
                await websocket.send_text(message_to_send)

            await asyncio.sleep(0.04)
    except WebSocketDisconnect:
        print("Dashboard UI client disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=video_loop, daemon=True)
    t.start()
    yield


app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
