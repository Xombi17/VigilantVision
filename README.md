# VigilantVision 🎥👁️
**Intelligent Video-Based Real-Time Retail Theft Detection & Surveillance System**

VigilantVision is a state-of-the-art, enterprise-grade video surveillance and anti-theft security solution designed for retail spaces, supermarkets, and smart facilities. It utilizes cutting-edge Computer Vision, real-time Pose Estimation, Object Detection, and custom-trained 3D-CNN (ResNet-3D) Action Classification to detect shoplifting, loitering, restricted area intrusions, and suspicious behaviors.

The system leverages optimized, multi-threaded pipelines to analyze concurrent camera feeds (local webcams or RTSP network cameras) synchronously, triggering instant browser-synthesized audio sirens and sending remote alerts via Email and Telegram.

---

## Team VigilantVision AI
*   **Varad Joshi** (Lead)
*   **Nathan Dsouza**
*   **Hrishikesh Nikam**
*   **Akshada Sapre**

---

## System Capabilities & Key Features

### 1. Multi-Threaded Camera Architecture
*   **Asynchronous Frame Reading:** Captures frames independently via high-performance python threading (`ThreadedCamera`), avoiding sequential frame capture lag or UI freezes.
*   **Robust Multi-Camera Tracking:** Dynamically resolves tracker ID conflicts across multiple feeds simultaneously by isolating states uniquely using camera-to-person composites `(camera_id, track_id)`.

### 2. 3D-CNN Action Classification (CLIP Model)
*   **3D ResNet-18 Action Classifier:** Real-time sliding window classification on a buffer of 16 video frames using the custom-trained PyTorch 3D-CNN model (`models/clip_classifier_best.pt`).
*   **Action Probability Alerts:** Instantly flags suspicious body language and motion patterns that indicate theft, providing confidence probability scores.

### 3. Advanced Behavior & Posture Estimation
*   **Item Concealment Logic:** Recognizes when a person picks up a target retail item and monitors hand-to-pocket/bag gestures, flagging potential concealment attempts.
*   **Loitering Detection:** Evaluates how long a person dwells within a specific ROI. If loitering exceeds the configurable threshold, an alarm is triggered.
*   **Zone Intrusion Alerts:** Instantly sounds sirens if human wrists cross into high-security zones (e.g., cash register areas, restricted aisles).
*   **Postural Suspicion:** Detects unusual physical behavior such as sudden bending down in low-visibility aisles.
*   **Activity Heatmaps:** Localized heatmap accumulators aggregate and visually plot customer traffic patterns individually for each camera stream.

### 4. Interactive Canvas ROI Drawer
*   **HTML5 Canvas Drawing Tool:** Draw precise security boundaries (Polygons) overlaying live webcam/RTSP feeds directly inside a glassmorphic dashboard modal.
*   **Resolution-Agnostic Scaling:** Autonomously maps client-side mouse vectors into exact `1280x720` surveillance matrix coordinates, preventing scaling discrepancies across different screen resolutions.
*   **Camera-Specific Storage:** Camera definitions and their respective ROI coordinate lists are saved persistently inside `cameras.json`.

### 5. Facial Recognition & Database Panel
*   **Face ID Classification:** A dedicated, premium **Face Management** panel to upload portrait photos, register new faces, and assign categorizations:
    *   **Blacklist:** Automatically triggers high-priority security alarms and records evidence.
    *   **VIP Whitelist:** Identifies trusted staff, loyal clients, or VIP visitors, showing a green greeting badge.
*   **Instant Face Database Deletion:** One-click instant SQLite deletion with automatic memory synchronization.

### 6. Client-Side Synthetic Audio Siren & Notifications
*   **Web Audio API Integration:** Avoids brittle MP3 loading loops by synthesizing realistic, sweeping emergency sirens directly in the browser's audio processor in real-time when alarms fire.
*   **High-Priority Cooldowns:** Protects users from noise fatigue by enforcing a 3-second smart alarm cooldown period.
*   **Telegram & SMTP Setup:** Instantly broadcast alerts and snapshots via Telegram chat integrations and automated Email notifications.

---

## Technical Architecture

*   **Backend Engine:** Python 3.10+, FastAPI (Asynchronous API endpoints & WebSockets), OpenCV (Multi-threaded streaming), Ultralytics YOLOv8 (Stand-alone Pose & Object model detection), PyTorch + Torchvision (3D ResNet-18 Action Classifier), `face_recognition` (Dlib-based CNN face encodings), SQLite3 (Database storage for logs & face matrices).
*   **Frontend Dashboard:** Next.js 14+ (App Router), React 18, Tailwind CSS, Recharts (Modern chart libraries), Lucide React (Fluent vector icons), HSL Custom Themes (Harmonious Glassmorphic Dark UI).

---

## Installation Guide

### Prerequisites
*   Python 3.9 - 3.11
*   Node.js (LTS version)
*   CUDA Enabled NVIDIA GPU (Highly recommended for fluid real-time inference)

### 1. Backend Configuration
Install the Python dependencies:

```bash
pip install -r requirements.txt
```

#### Required Models
The system uses the following pre-trained models:
- `models/clip_classifier_best.pt` — Custom-trained 3D-CNN (ResNet-3D) action classifier.
- `yolov8n.pt` — Object detection (person tracking, item monitoring)
- `yolov8n-pose.pt` — Pose estimation (posture analysis, gesture detection)

### 2. Dashboard UI Configuration
Install node packages:

```bash
cd dashboard
npm install
```

### 3. Verify Installation
Run the setup tests to confirm everything is working:

```bash
python test_setup.py      # Tests OpenCV + YOLO object detection
python test_pose.py       # Tests YOLO pose model loading
```

---

## Running the System

### Automatic Startup
Launch both the FastAPI service and the Next.js development server concurrently:

```bash
start_system.bat
```

### Manual Startup
**1. Start the API Server & Inference Loop:**
```bash
python backend.py
```
*(The server will boot on `http://localhost:8000` — Swagger docs at `http://localhost:8000/docs`)*

**2. Start the Frontend Dashboard:**
```bash
cd dashboard
npm run dev
```
*(The panel will be served on `http://localhost:3000`)*

**3. Optional Standalone OpenCV Window Demo:**
```bash
python standalone_demo.py
```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
