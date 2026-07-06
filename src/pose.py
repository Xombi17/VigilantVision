import cv2
import numpy as np


# --- State Tracker for Concealment ---
class PersonState:
    def __init__(self, track_id):
        self.track_id = track_id
        self.state = "NEUTRAL"  # NEUTRAL, REACHING, HOLDING, SUSPICIOUS
        self.last_reach_time = 0
        self.holding_object = False
        self.holding_hand = None
        self.last_holding_time = 0
        self.face_checked = False
        self.face_check_time = 0
        self.is_vip = False


# Global registry of person states
person_states = {}  # {(cam_id, track_id): PersonState}


# --- Helper Functions for Pose ---
def check_reaching(keypoints, roi_poly):
    if len(keypoints) < 11:
        return False, None
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    reaching_hand = None

    if left_wrist[0] > 0 and left_wrist[1] > 0 and len(roi_poly) >= 3:
        if (
            cv2.pointPolygonTest(
                np.array(roi_poly), (int(left_wrist[0]), int(left_wrist[1])), False
            )
            >= 0
        ):
            reaching_hand = "LEFT"

    if right_wrist[0] > 0 and right_wrist[1] > 0 and len(roi_poly) >= 3:
        if (
            cv2.pointPolygonTest(
                np.array(roi_poly), (int(right_wrist[0]), int(right_wrist[1])), False
            )
            >= 0
        ):
            reaching_hand = "RIGHT"

    return (reaching_hand is not None), reaching_hand


def check_object_in_hand(keypoints, object_boxes, hand="LEFT"):
    if len(keypoints) < 11:
        return False
    wrist = keypoints[9] if hand == "LEFT" else keypoints[10]

    if wrist[0] == 0:
        return False

    for box in object_boxes:
        box_cx = (box[0] + box[2]) / 2
        box_cy = (box[1] + box[3]) / 2

        dist = np.sqrt((wrist[0] - box_cx) ** 2 + (wrist[1] - box_cy) ** 2)

        if dist < 120:  # Threshold
            return True
        if box[0] < wrist[0] < box[2] and box[1] < wrist[1] < box[3]:
            return True

    return False


def check_concealment(keypoints, reaching_hand):
    if len(keypoints) < 13:
        return False
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    target_wrist = keypoints[9] if reaching_hand == "LEFT" else keypoints[10]

    if target_wrist[0] == 0 or left_hip[0] == 0 or right_hip[0] == 0:
        return False

    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2

    dist_x = target_wrist[0] - hip_center_x
    dist_y = target_wrist[1] - hip_center_y
    distance = np.sqrt(dist_x**2 + dist_y**2)

    hip_width = np.abs(left_hip[0] - right_hip[0])
    threshold = max(hip_width * 1.5, 100)

    return distance < threshold


def check_bending(keypoints):
    if len(keypoints) < 12:
        return False
    l_shoulder = keypoints[5]
    l_hip = keypoints[11]
    if l_shoulder[1] == 0 or l_hip[1] == 0:
        return False
    vertical_dist = l_hip[1] - l_shoulder[1]
    return vertical_dist < 50


# --- Heatmap Logic ---
def update_heatmap(cam_data, center_x, center_y, frame_shape):
    if (
        cam_data.get("heatmap_accumulator") is None
        or cam_data["heatmap_accumulator"].shape[:2] != frame_shape[:2]
    ):
        cam_data["heatmap_accumulator"] = np.zeros(frame_shape[:2], dtype=np.float32)
    try:
        cam_data["heatmap_accumulator"][center_y, center_x] += 1
    except:
        pass


def get_heatmap_overlay(cam_data, frame):
    if cam_data.get("heatmap_accumulator") is None:
        return frame
    msg_max = np.max(cam_data["heatmap_accumulator"])
    if msg_max == 0:
        return frame

    norm_heatmap = cam_data["heatmap_accumulator"] / msg_max
    norm_heatmap = (norm_heatmap * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
    result = cv2.addWeighted(frame, 0.7, color_map, 0.3, 0)
    return result
