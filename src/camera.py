import cv2
import threading
import time
import json
import os
import uuid


class ThreadedCamera:
    def __init__(self, src):
        self.src = src
        try:
            self.src_val = int(src)
            is_index = True
        except:
            self.src_val = src
            is_index = False

        if is_index and os.name == "nt":
            self.cap = cv2.VideoCapture(self.src_val, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.src_val)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.ret, self.frame = self.cap.read()
        else:
            self.ret = False
            self.frame = None

        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        if self.cap.isOpened():
            self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                # If reading fails and we are streaming from a video file, rewind and loop it
                if not ret and not isinstance(self.src_val, int):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                with self.lock:
                    self.ret = ret
                    if ret:
                        self.frame = frame
                time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        self.cap.release()


class CameraManager:
    def __init__(self):
        self.cameras = {}
        self.lock = threading.Lock()
        self.load_cameras()

    def load_cameras(self):
        file_path = "cameras.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for cam in data:
                        self.add_camera_internal(
                            cam["id"],
                            cam["source"],
                            cam["name"],
                            cam.get("roi_points", []),
                        )
                print(f"Loaded {len(self.cameras)} cameras from cameras.json.")
                return
            except Exception as e:
                print(f"Error loading cameras.json: {e}")

        # Fallback to default webcam if no file exists
        self.add_camera_internal("0", "0", "Camera 1", [])
        self.save_cameras()

    def save_cameras(self):
        file_path = "cameras.json"
        try:
            data = []
            with self.lock:
                for cam_id, cam_data in self.cameras.items():
                    data.append(
                        {
                            "id": cam_id,
                            "name": cam_data["name"],
                            "source": cam_data["source"],
                            "roi_points": cam_data.get("roi_points", []),
                        }
                    )
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving cameras.json: {e}")

    def add_camera_internal(self, cam_id, source, name, roi_points):
        threaded_cap = ThreadedCamera(source)
        self.cameras[cam_id] = {
            "cap": threaded_cap,
            "name": name,
            "source": source,
            "status": "active" if threaded_cap.isOpened() else "error",
            "roi_points": roi_points,
            "heatmap_accumulator": None,
            "roi_entry_times": {},
            "last_alert_time": 0,
        }

    def add_camera(self, source, name):
        cam_id = str(uuid.uuid4())
        threaded_cap = ThreadedCamera(source)
        if threaded_cap.isOpened():
            with self.lock:
                self.cameras[cam_id] = {
                    "cap": threaded_cap,
                    "name": name,
                    "source": source,
                    "status": "active",
                    "roi_points": [],
                    "heatmap_accumulator": None,
                    "roi_entry_times": {},
                    "last_alert_time": 0,
                }
            self.save_cameras()
            print(f"Camera added: {name} ({source}) ID: {cam_id}")
            return {"id": cam_id, "status": "connected"}
        else:
            print(f"Failed to open camera: {source}")
            return {"id": None, "status": "failed"}

    def remove_camera(self, cam_id):
        with self.lock:
            if cam_id in self.cameras:
                self.cameras[cam_id]["cap"].release()
                del self.cameras[cam_id]
                status = True
            else:
                status = False
        if status:
            self.save_cameras()
        return status

    def get_active_cameras(self):
        with self.lock:
            return [
                {
                    "id": k,
                    "name": v["name"],
                    "source": v["source"],
                    "status": "active" if v["cap"].isOpened() else "error",
                    "roi_points": v.get("roi_points", []),
                }
                for k, v in self.cameras.items()
            ]


camera_manager = CameraManager()
