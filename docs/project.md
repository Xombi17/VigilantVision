VigilantVision
Intelligent Video-Based Real-Time Retail Theft Detection and Alert System
Project Synopsis — Team VigilantVision AI

Team Details

Team Name: VigilantVision AI

Role

Member

Team Lead / Contributor

Varad Joshi

Contributor

Contributor

Contributor

Abstract

Nathan Dsouza

Hrishikesh Nikam

Akshadha Sapre

VigilantVision is a proactive, AI-driven video surveillance system designed to identify retail theft and shoplifting actions in
real time. Unlike traditional passive surveillance systems that merely record incidents for forensic review, VigilantVision
actively monitors customer and product interactions as they occur.

The system uses high-performance computer vision pipelines, including real-time deep learning object detection, multi-object
tracking, and pose estimation. By extracting spatial skeletal coordinates and evaluating geometric relationships between
customers, products, and shopping carts, it identifies suspicious gestures such as item concealment. When a high-probability
theft anomaly is detected, an instant alert with a short visual log is pushed to a centralized, web-based monitoring dashboard,
enabling rapid on-site intervention before a suspect exits.

Objectives

•  Develop a real-time human, inventory, and cart detection and tracking system using state-of-the-art deep learning

architectures.

•

Implement a gesture and posture analysis module using video-based spatial skeletal mapping.

•  Build a high-performance heuristic state machine capable of evaluating interaction anomalies while minimizing

false positives.

•  Design a centralized web dashboard to display live monitoring feeds, warning notifications, and security logs.

•  Optimize model execution to guarantee real-time performance on standard computing hardware.

System Architecture Overview

The pipeline consists of four integrated layers:

1. Input Layer

Ingestion of real-time camera streams (RTSP network streams or local webcam), processed sequentially via optimized
OpenCV frames.

2. Processing Layer

Deep learning inference using YOLOv8/YOLOv11 for localized detection of shoppers, items, and carts, combined with
ByteTrack for persistent ID tracking across temporal occlusions. Concurrently, YOLO-Pose extracts 17-point skeletal
coordinate arrays for gesture analysis.

3. Logic Layer

A geometric state engine tracks the Euclidean distance between hand keypoints and target product bounding boxes over a
temporal sliding frame window. An anomaly is flagged if a product bounding box disappears within a person's torso region
without passing through a shopping cart's boundary.

4. Application Layer

A lightweight FastAPI web dashboard presenting live feeds, tracking metrics, real-time alert widgets, and a persistent SQLite
incident log.

Methodology

•  Video frames are ingested and pre-processed dynamically using OpenCV.

•  A pre-trained YOLO detector locates people, shopping carts, and high-value objects.

•  ByteTrack associates detections across frames to maintain stable tracking identities even during partial or temporary

visual occlusion.

•

Frame-by-frame pose estimation maps hand joint coordinates of identified targets relative to surrounding item
bounding boxes.

•  A spatial logic engine evaluates proximity: if an object bounding box is occluded while in contact with a subject's

hand and fails to register inside a cart, a suspected concealment state is triggered.

•  Triggered anomalies produce immediate alert metadata, pushed directly to the FastAPI-served monitoring panel and

stored locally.

Technology Stack

Category

Technology

Programming Language

Python 3.10+

Backend Web Framework

FastAPI

Computer Vision Frameworks

OpenCV, Ultralytics YOLO (Object & Pose models)

Object Tracker

ByteTrack

Optimization Frameworks

ONNX Runtime / TensorRT

Frontend Web Stack

HTML5, CSS3, JavaScript

Database & Logging

SQLite / JSON / supabase / neon db / mongo db

Containerization

Docker

Implementation Roadmap (8-Day Plan)

Day

Focus

Key Tasks

Day 1  Workspace Init & Stream

Ingestion

Set up Python 3.10+, PyTorch, OpenCV environment. Configure
webcam/RTSP ingestion and load simulated feeds from the
UCF-Crime Shoplifting dataset.

Day 2

Object Detection & ByteTrack
Assembly

Integrate YOLO to detect person, backpack, bottle, and cart classes.
Bind ByteTrack to assign and persist unique tracking IDs.

Day 3

Skeletal Joint Estimation

Deploy YOLO-Pose to extract skeletal keypoints for all active
person targets, with focus on left/right wrist coordinates.

Day 4

Day 5

Spatial Distance Algorithm &
Proximity Maps

Implement Euclidean distance routines between wrist keypoints and
item bounding boxes; map hand-to-item intersections.

Behavioral State Machine &
Concealment Logic

Build state transitions flagging products that leave visibility and
enter torso/pocket regions without passing through the cart or
register zone.

Day 6

FastAPI Dashboard & Live
Warning Feed

Build the FastAPI backend with WebSocket/streaming endpoints.
Create a responsive HTML/CSS/JS frontend to stream annotated
frames and display real-time alert popups.

Day 7

Day 8

Performance Profiling &
ONNX Optimization

Export weights to ONNX, quantize to FP16, and benchmark FPS
against the real-time deployment threshold on standard hardware.

Integration Testing & Docker
Packaging

Run full pipeline tests with simulated shoplifting scenarios. Package
the platform and FastAPI server into a lightweight Docker container.

Expected Outcome

A modular, high-efficiency retail surveillance prototype capable of running end-to-end video analytics. The system will
demonstrate real-time person-to-object tracking, recognize concealment gestures with a low margin of error, and deliver
visual alert feeds to a centralized security dashboard within a near-instant threshold of the physical incident.

Feasibility and Scope

By leveraging optimized pre-trained architectures (YOLO + ByteTrack) for low-level perception and lightweight geometric
distance calculations for high-level decision logic, the pipeline avoids the intense compute demands of training heavy,
end-to-end 3D Convolutional Neural Networks (3D-CNNs).

This design makes the system highly feasible on standard, non-specialized consumer hardware, such as laptops with mid-tier
local GPUs. The modular layout also ensures straightforward scalability — permitting future updates such as multi-camera
RTSP ingestion, centralized database integration, and low-power edge deployment.


