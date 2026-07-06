import os
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO

from src.config import ALERT_COOLDOWN, LOITERING_THRESHOLD, FACE_REC_AVAILABLE
from src.database import (
    known_face_encodings,
    known_face_names,
    known_face_types,
    faces_lock,
)
from src.pose import (
    PersonState,
    person_states,
    check_reaching,
    check_object_in_hand,
    check_concealment,
    check_bending,
    update_heatmap,
    get_heatmap_overlay,
)
from src.alerts import trigger_alert

if FACE_REC_AVAILABLE:
    import face_recognition


class SurveillanceEngine:
    def __init__(self):
        print("Initializing SurveillanceEngine models...")
        self.model_pose = None
        self.model_obj = None
        self.model_is_specialized = False

        self.clip_model = None
        self.clip_device = "cpu"
        self.clip_buffers = {}

        self.load_models()

    def load_models(self):
        try:
            print("Loading Pose Model (yolov8n-pose.pt)...")
            self.model_pose = YOLO("yolov8n-pose.pt")

            print("Loading VigilantVision Object Model...")
            try:
                self.model_obj = YOLO("shoplifting.pt")
                self.model_is_specialized = True
                print("Special Shoplifting Model Loaded! (shoplifting.pt)")
            except:
                print(
                    "Special model not found, falling back to standard object tracking (yolov8n.pt)..."
                )
                try:
                    self.model_obj = YOLO("yolov8n.pt")
                except Exception as e:
                    print(f"Failed to load standard model: {e}")
                    self.model_obj = None

            print("Models ready.")

            # Load PyTorch 3D-CNN (CLIP) classifier
            try:
                import torch
                from torchvision.models.video import r3d_18

                clip_path = "models/clip_classifier_best.pt"
                if os.path.exists(clip_path):
                    print(
                        "Loading PyTorch 3D-CNN (CLIP) classifier from models/clip_classifier_best.pt..."
                    )
                    self.clip_device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.clip_model = r3d_18(weights=None)
                    self.clip_model.fc = torch.nn.Linear(
                        self.clip_model.fc.in_features, 1
                    )
                    self.clip_model.load_state_dict(
                        torch.load(clip_path, map_location=self.clip_device)
                    )
                    self.clip_model.to(self.clip_device)
                    self.clip_model.eval()
                    print(f"PyTorch 3D-CNN loaded successfully on {self.clip_device}!")
            except Exception as e:
                print(f"Could not load PyTorch 3D-CNN (CLIP) classifier: {e}")

        except Exception as e:
            print(f"CRITICAL MODEL ERROR: {e}")
            with open("error_log.txt", "a") as f:
                f.write(f"{datetime.now()}: CRITICAL LOAD ERROR: {e}\n")

    def process_frame(self, cam_id, name, frame, cam_data, frame_count):
        import torch

        current_time = time.time()
        cam_roi = cam_data.get("roi_points", [])

        # --- 3D-CNN CLIP Classifier Inference ---
        clip_confidence = cam_data.get("last_clip_confidence", 0.0)
        if self.clip_model is not None:
            try:
                # Check if any tracked person on this camera is currently identified as VIP
                vip_present = False
                for state_key, p_state_check in person_states.items():
                    if state_key[0] == cam_id and getattr(
                        p_state_check, "is_vip", False
                    ):
                        vip_present = True
                        break

                if cam_id not in self.clip_buffers:
                    self.clip_buffers[cam_id] = []
                resized_f = cv2.resize(frame, (112, 112))
                rgb_f = cv2.cvtColor(resized_f, cv2.COLOR_BGR2RGB)
                self.clip_buffers[cam_id].append(rgb_f)
                if len(self.clip_buffers[cam_id]) > 16:
                    self.clip_buffers[cam_id].pop(0)

                if len(self.clip_buffers[cam_id]) == 16 and (frame_count % 10 == 0):
                    clip_tensor = (
                        torch.tensor(
                            np.array(self.clip_buffers[cam_id]), dtype=torch.float32
                        )
                        / 255.0
                    )
                    clip_tensor = clip_tensor.permute(0, 3, 1, 2)

                    mean_val = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
                    std_val = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
                    clip_tensor = (clip_tensor - mean_val) / std_val

                    clip_tensor = clip_tensor.permute(1, 0, 2, 3).unsqueeze(0)
                    clip_tensor = clip_tensor.to(self.clip_device)

                    with torch.no_grad():
                        logits = self.clip_model(clip_tensor).squeeze(1)
                        clip_confidence = float(torch.sigmoid(logits).item())
                        cam_data["last_clip_confidence"] = clip_confidence

                if clip_confidence > 0.6 and not vip_present:
                    cv2.putText(
                        frame,
                        f"THEFT DETECTED (3D-CNN: {clip_confidence:.2f})",
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )
                    if current_time - cam_data["last_alert_time"] > ALERT_COOLDOWN:
                        trigger_alert(
                            cam_id,
                            name,
                            f"THEFT SUSPICION (3D-CNN Conf: {clip_confidence:.2f})",
                            frame,
                        )
                        cam_data["last_alert_time"] = current_time
            except Exception as e:
                print(f"3D-CNN inference error: {e}")

        # --- YOLO Object detection ---
        run_obj_det = (frame_count % 5 == 0) and (self.model_obj is not None)
        detected_objects = []

        if run_obj_det:
            if self.model_is_specialized:
                results_obj = self.model_obj(frame, verbose=False, conf=0.45)
                if len(results_obj) > 0:
                    boxes_obj = results_obj[0].boxes.xyxy.cpu().numpy().astype(int)
                    cls_obj = results_obj[0].boxes.cls.cpu().numpy().astype(int)
                    conf_obj = results_obj[0].boxes.conf.cpu().numpy()

                    for b, c, conf in zip(boxes_obj, cls_obj, conf_obj):
                        class_name = self.model_obj.names[c]
                        detected_objects.append(b)
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"{class_name} {conf:.2f}",
                            (b[0], b[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                        )

                        if class_name in ["shoplift", "theft", "suspicious", "fight"]:
                            if (
                                current_time - cam_data["last_alert_time"]
                                > ALERT_COOLDOWN
                            ):
                                trigger_alert(
                                    cam_id,
                                    name,
                                    f"CRIMINAL ACTIVITY: {class_name}",
                                    frame,
                                )
                                cam_data["last_alert_time"] = current_time
                        else:
                            cv2.rectangle(
                                frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1
                            )
            else:
                TARGET_CLASSES = [
                    24,
                    25,
                    26,
                    28,
                    39,
                    40,
                    41,
                    42,
                    43,
                    67,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                ]
                results_obj = self.model_obj(frame, verbose=False, conf=0.3)
                if len(results_obj) > 0:
                    boxes_obj = results_obj[0].boxes.xyxy.cpu().numpy().astype(int)
                    cls_obj = results_obj[0].boxes.cls.cpu().numpy().astype(int)
                    conf_obj = results_obj[0].boxes.conf.cpu().numpy()

                    for b, c, conf in zip(boxes_obj, cls_obj, conf_obj):
                        if c in TARGET_CLASSES:
                            detected_objects.append(b)
                            label = f"ITEM: {self.model_obj.names[c]} {conf:.2f}"
                            cv2.rectangle(
                                frame, (b[0], b[1]), (b[2], b[3]), (0, 165, 255), 2
                            )
                            cv2.putText(
                                frame,
                                label,
                                (b[0], b[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 165, 255),
                                1,
                            )

        if run_obj_det:
            cam_data["last_objects"] = detected_objects
        else:
            detected_objects = cam_data.get("last_objects", [])

        # --- YOLOv8-Pose skeletal loops ---
        results_pose = self.model_pose.track(
            frame, persist=True, verbose=False, classes=[0]
        )

        if results_pose[0].boxes.id is not None:
            boxes = results_pose[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results_pose[0].boxes.id.cpu().numpy().astype(int)

            try:
                keypoints_all = results_pose[0].keypoints.xy.cpu().numpy()
            except:
                keypoints_all = []

            for i, track_id in enumerate(track_ids):
                box = boxes[i]
                kpts = keypoints_all[i] if len(keypoints_all) > i else []

                state_key = (cam_id, track_id)
                if state_key not in person_states:
                    person_states[state_key] = PersonState(track_id)
                p_state = person_states[state_key]

                is_bending = False
                is_reaching = False

                # --- Face ID Match checks ---
                if FACE_REC_AVAILABLE and (
                    not p_state.face_checked
                    or (current_time - p_state.face_check_time > 2.0)
                ):
                    p_state.face_check_time = current_time
                    fx1, fy1, fx2, fy2 = (
                        max(0, box[0]),
                        max(0, box[1]),
                        min(frame.shape[1], box[2]),
                        min(frame.shape[0], box[3]),
                    )
                    face_img = frame[fy1:fy2, fx1:fx2]
                    rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_locs = face_recognition.face_locations(rgb_face)
                    if face_locs:
                        encodings = face_recognition.face_encodings(rgb_face, face_locs)
                        if encodings:
                            with faces_lock:
                                matches = face_recognition.compare_faces(
                                    known_face_encodings, encodings[0], tolerance=0.5
                                )
                            if True in matches:
                                match_index = matches.index(True)
                                match_name = known_face_names[match_index]
                                match_type = known_face_types[match_index]
                                if match_type == "blacklist":
                                    p_state.is_vip = False
                                    cv2.putText(
                                        frame,
                                        f"BLACKLIST: {match_name}",
                                        (box[0], box[1] - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 0, 255),
                                        3,
                                    )
                                    if (
                                        current_time - cam_data["last_alert_time"]
                                        > ALERT_COOLDOWN
                                    ):
                                        trigger_alert(
                                            cam_id,
                                            name,
                                            f"BLACKLIST FACE: {match_name}",
                                            frame,
                                        )
                                        cam_data["last_alert_time"] = current_time
                                else:
                                    cv2.putText(
                                        frame,
                                        f"VIP: {match_name}",
                                        (box[0], box[1] - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1,
                                        (0, 255, 0),
                                        2,
                                    )
                                    p_state.is_vip = True
                            else:
                                p_state.is_vip = False
                        else:
                            p_state.is_vip = False
                    else:
                        p_state.is_vip = False
                    p_state.face_checked = True

                # --- Posture Heuristics ---
                is_bending = check_bending(kpts)

                if not self.model_is_specialized:
                    left_has_obj = check_object_in_hand(kpts, detected_objects, "LEFT")
                    right_has_obj = check_object_in_hand(
                        kpts, detected_objects, "RIGHT"
                    )
                    current_holding = left_has_obj or right_has_obj
                    holding_hand = (
                        "LEFT" if left_has_obj else "RIGHT" if right_has_obj else None
                    )

                    if current_holding:
                        p_state.holding_object = True
                        p_state.last_holding_time = current_time
                        p_state.holding_hand = holding_hand
                        if not getattr(p_state, "is_vip", False):
                            cv2.putText(
                                frame,
                                f"HOLDING ({holding_hand})",
                                (box[0], box[1] - 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 0),
                                2,
                            )

                    if p_state.holding_object and not current_holding:
                        time_since_hold = current_time - p_state.last_holding_time
                        if time_since_hold < 3.0:
                            hand_to_check = p_state.holding_hand
                            if hand_to_check and check_concealment(kpts, hand_to_check):
                                if not getattr(p_state, "is_vip", False):
                                    cv2.putText(
                                        frame,
                                        "THEFT DETECTED!",
                                        (box[0], box[1] - 80),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2,
                                        (0, 0, 255),
                                        3,
                                    )
                                    cv2.rectangle(
                                        frame,
                                        (box[0], box[1]),
                                        (box[2], box[3]),
                                        (0, 0, 255),
                                        3,
                                    )
                                    if (
                                        current_time - cam_data["last_alert_time"]
                                        > ALERT_COOLDOWN
                                    ):
                                        trigger_alert(
                                            cam_id,
                                            name,
                                            "THEFT CONFIRMED (Item Concealed)",
                                            frame,
                                        )
                                        cam_data["last_alert_time"] = current_time
                                        p_state.holding_object = False
                        else:
                            if time_since_hold > 3.0:
                                p_state.holding_object = False
                                p_state.holding_hand = None

                # --- ROI check ---
                is_reaching, _ = check_reaching(kpts, cam_roi)
                if is_reaching:
                    if not getattr(p_state, "is_vip", False):
                        cv2.putText(
                            frame,
                            "RESTRICTED AREA ENT!",
                            (box[0], box[1] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        if current_time - cam_data["last_alert_time"] > ALERT_COOLDOWN:
                            trigger_alert(
                                cam_id, name, "RESTRICTED AREA INTRUSION", frame
                            )
                            cam_data["last_alert_time"] = current_time

                if is_bending:
                    if not getattr(p_state, "is_vip", False):
                        cv2.putText(
                            frame,
                            "BENDING",
                            (box[0], box[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2,
                        )

                # --- Heatmaps & Loitering durations ---
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                update_heatmap(cam_data, center_x, center_y, frame.shape)

                is_inside_roi = False
                if len(cam_roi) >= 3:
                    if (
                        cv2.pointPolygonTest(
                            np.array(cam_roi), (center_x, center_y), False
                        )
                        >= 0
                    ):
                        is_inside_roi = True

                if is_inside_roi:
                    if track_id not in cam_data["roi_entry_times"]:
                        cam_data["roi_entry_times"][track_id] = time.time()
                    duration = time.time() - cam_data["roi_entry_times"][track_id]
                    if not getattr(p_state, "is_vip", False):
                        cv2.putText(
                            frame,
                            f"{duration:.1f}s",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2,
                        )

                        if duration > LOITERING_THRESHOLD:
                            if (
                                current_time - cam_data["last_alert_time"]
                                > ALERT_COOLDOWN
                            ):
                                trigger_alert(
                                    cam_id, name, "LOITERING SUSPICION", frame
                                )
                                cam_data["last_alert_time"] = current_time
                else:
                    if track_id in cam_data["roi_entry_times"]:
                        del cam_data["roi_entry_times"][track_id]

        frame = get_heatmap_overlay(cam_data, frame)

        if results_pose[0].keypoints is not None:
            res_plotted = results_pose[0].plot()
            frame = res_plotted

        if len(cam_roi) > 0:
            cv2.polylines(
                frame,
                [np.array(cam_roi)],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )

        return frame
