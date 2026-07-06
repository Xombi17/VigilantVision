import cv2
from ultralytics import YOLO
import numpy as np
import time
import os
import argparse
from datetime import datetime

# Parse arguments
parser = argparse.ArgumentParser(description="VigilantVision Standalone Video Demo")
parser.add_argument(
    "--source",
    type=str,
    default="0",
    help="Webcam index (e.g. 0) or path to video file",
)
parser.add_argument(
    "--mirror", action="store_true", help="Mirror flip the video feed horizontally"
)
args = parser.parse_args()

# Try to parse source as integer for webcam
try:
    source_val = int(args.source)
    is_webcam = True
except ValueError:
    source_val = args.source
    is_webcam = False

# Determine if we should mirror
should_mirror = args.mirror or (is_webcam and not args.mirror)

# Global variables for ROI
roi_points = []
drawing = False

# Dictionary to store entry times for IDs in ROI: {track_id: start_time}
roi_entry_times = {}
LOITERING_THRESHOLD = 5.0
last_alert_time = 0
ALERT_COOLDOWN = 3.0

# Create alerts directory
if not os.path.exists("alerts"):
    os.makedirs("alerts")


def mouse_callback(event, x, y, flags, param):
    global roi_points, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        roi_points = []
        drawing = False


def is_inside_roi(point, polygon):
    if len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(np.array(polygon), point, False) >= 0


def check_bending(keypoints):
    if len(keypoints) < 12:
        return False
    l_shoulder = keypoints[5]
    l_hip = keypoints[11]
    if l_shoulder[1] == 0 or l_hip[1] == 0:
        return False
    vertical_dist = l_hip[1] - l_shoulder[1]
    return vertical_dist < 50


def check_reaching(keypoints, roi_poly):
    if len(keypoints) < 11:
        return False
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    if left_wrist[0] > 0 and left_wrist[1] > 0 and len(roi_poly) >= 3:
        if (
            cv2.pointPolygonTest(
                np.array(roi_poly), (int(left_wrist[0]), int(left_wrist[1])), False
            )
            >= 0
        ):
            return True

    if right_wrist[0] > 0 and right_wrist[1] > 0 and len(roi_poly) >= 3:
        if (
            cv2.pointPolygonTest(
                np.array(roi_poly), (int(right_wrist[0]), int(right_wrist[1])), False
            )
            >= 0
        ):
            return True

    return False


def main():
    global last_alert_time, roi_points

    # Load YOLOv8 models
    print("Loading Pose Model (yolov8n-pose)...")
    model_pose = YOLO("yolov8n-pose.pt")

    print("Loading Tracking Model (yolov8n)...")
    model_track = YOLO("yolov8n.pt")

    # Load 3D-CNN (CLIP) classifier
    clip_model = None
    clip_device = "cpu"
    clip_buffer = []
    last_clip_confidence = 0.0

    try:
        import torch
        from torchvision.models.video import r3d_18

        clip_path = "models/clip_classifier_best.pt"
        if os.path.exists(clip_path):
            print(f"Loading PyTorch 3D-CNN classifier from {clip_path}...")
            clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model = r3d_18(weights=None)
            clip_model.fc = torch.nn.Linear(clip_model.fc.in_features, 1)
            clip_model.load_state_dict(torch.load(clip_path, map_location=clip_device))
            clip_model.to(clip_device)
            clip_model.eval()
            print(f"PyTorch 3D-CNN loaded successfully on {clip_device}!")
        else:
            print(
                "PyTorch 3D-CNN classifier model not found. Action classification disabled."
            )
    except Exception as e:
        print(f"Could not load PyTorch 3D-CNN: {e}")

    # Open the video source
    print(f"Opening video source: {args.source}...")
    if is_webcam and os.name == "nt":
        cap = cv2.VideoCapture(source_val, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"Error: Could not open source {args.source}")
        return

    # Set resolution
    width = 1280
    height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Setup mouse callback
    window_name = "VigilantVision Standalone System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\nSystem ready.")
    print("- Left click: Add restricted area (ROI) point.")
    print("- Right click: Clear restricted area.")
    print("- 'q': Exit.")

    frame_count = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()

        # Auto-loop video files
        if not ret:
            if not is_webcam:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break

        # Mirror flip if enabled
        if should_mirror:
            frame = cv2.flip(frame, 1)

        # 1. Run 3D-CNN Action Classification (once every 10 frames)
        if clip_model is not None:
            try:
                resized_f = cv2.resize(frame, (112, 112))
                rgb_f = cv2.cvtColor(resized_f, cv2.COLOR_BGR2RGB)
                clip_buffer.append(rgb_f)
                if len(clip_buffer) > 16:
                    clip_buffer.pop(0)

                if len(clip_buffer) == 16 and (frame_count % 10 == 0):
                    clip_tensor = (
                        torch.tensor(np.array(clip_buffer), dtype=torch.float32) / 255.0
                    )
                    clip_tensor = clip_tensor.permute(0, 3, 1, 2)

                    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1)
                    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1)
                    clip_tensor = (clip_tensor - mean) / std
                    clip_tensor = (
                        clip_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(clip_device)
                    )

                    with torch.no_grad():
                        logits = clip_model(clip_tensor).squeeze(1)
                        last_clip_confidence = float(torch.sigmoid(logits).item())
            except Exception as e:
                print(f"3D-CNN inference error: {e}")

        # 2. Run YOLOv8 Tracking & Pose
        results_pose = model_pose.track(frame, persist=True, verbose=False, classes=[0])
        results_track = model_track.track(
            frame, persist=True, verbose=False, classes=[0]
        )

        current_ids_in_roi = set()
        detection_alert = False
        alert_message = ""

        # Draw ROI Boundary
        if len(roi_points) > 0:
            cv2.polylines(
                frame,
                [np.array(roi_points)],
                isClosed=True,
                color=(0, 255, 255),
                thickness=2,
            )
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(roi_points)], (0, 255, 255))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Process Pose keypoints
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

                # Check posture features
                is_bending = check_bending(kpts)
                is_reaching = check_reaching(kpts, roi_points)

                center_x = int((box[0] + box[2]) / 2)
                center_y = int(box[3])

                # ROI Checks
                inside = is_inside_roi((center_x, center_y), roi_points)
                color = (0, 255, 0)
                status = "Normal"

                if inside:
                    current_ids_in_roi.add(track_id)
                    if track_id not in roi_entry_times:
                        roi_entry_times[track_id] = time.time()
                    duration = time.time() - roi_entry_times[track_id]

                    if duration > LOITERING_THRESHOLD:
                        color = (0, 0, 255)
                        status = "SUSPICIOUS (LOITERING)!"
                        detection_alert = True
                        alert_message = (
                            f"ID:{track_id} loitering inside ROI ({duration:.1f}s)"
                        )
                    else:
                        color = (0, 165, 255)
                        status = f"Warning ({duration:.1f}s)"
                else:
                    if track_id in roi_entry_times:
                        roi_entry_times.pop(track_id)

                if is_reaching:
                    color = (0, 0, 255)
                    status = "RESTRICTED AREA ENTRY!"
                    detection_alert = True
                    alert_message = f"ID:{track_id} crossed ROI security line!"

                if is_bending:
                    cv2.putText(
                        frame,
                        "BENDING",
                        (box[0], box[3] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(
                    frame,
                    f"Person {track_id} ({status})",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # Plot pose skeleton
        if results_pose[0].keypoints is not None:
            frame = results_pose[0].plot()

        # Handle 3D-CNN alerts
        if last_clip_confidence > 0.6:
            detection_alert = True
            alert_message = (
                f"Theft pattern detected by 3D-CNN (Conf: {last_clip_confidence:.2f})"
            )
            cv2.putText(
                frame,
                f"THEFT DETECTED (3D-CNN: {last_clip_confidence:.2f})",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )

        # Alert cooldown & image snapshot saving
        if detection_alert and (time.time() - last_alert_time > ALERT_COOLDOWN):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alerts/alert_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Alert Triggered: {alert_message} | Image saved: {filename}")
            last_alert_time = time.time()

        # Render general warning on top of the frame
        if detection_alert:
            cv2.putText(
                frame,
                "SUSPICIOUS ACTIVITY DETECTED!",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

        # Show FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Resize frame to a comfortable size for viewing if it is too small
        h, w = frame.shape[:2]
        target_width = 1024
        if w < target_width:
            scale = target_width / w
            target_height = int(h * scale)
            display_frame = cv2.resize(frame, (target_width, target_height))
        else:
            display_frame = frame

        cv2.imshow(window_name, display_frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
