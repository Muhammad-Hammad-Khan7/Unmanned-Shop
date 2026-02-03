# real_time_sync_capture.py

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from threading import Thread
from queue import Queue
import time

# -----------------------------
# Camera IPs
# -----------------------------
CAMERA_1_IP = "rtsp://admin:SA%40112233@200.10.15.64:554/Streaming/Channels/101"
CAMERA_2_IP = "rtsp://admin:SA%40112233@200.10.15.65:554/Streaming/Channels/101"

# -----------------------------
# Save directories
# -----------------------------
root_dir = Path("data")
frame_dir = root_dir / "frames"
joint_dir = root_dir / "joints"

cam1_frame_dir = frame_dir / "cam1"
cam2_frame_dir = frame_dir / "cam2"
cam1_joint_dir = joint_dir / "cam1"
cam2_joint_dir = joint_dir / "cam2"

for d in [cam1_frame_dir, cam2_frame_dir, cam1_joint_dir, cam2_joint_dir]:
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------
# MediaPipe Pose setup
# -----------------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# -----------------------------
# Queues for thread-safe capture
# -----------------------------
queue1 = Queue(maxsize=5)
queue2 = Queue(maxsize=5)

# -----------------------------
# Camera capture threads
# -----------------------------
def camera_reader(cam_ip, q, cam_name):
    cap = cv2.VideoCapture(cam_ip)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {cam_name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = time.time()
        if not q.full():
            q.put((timestamp, frame))
    cap.release()

# Start camera threads
t1 = Thread(target=camera_reader, args=(CAMERA_1_IP, queue1, "cam1"), daemon=True)
t2 = Thread(target=camera_reader, args=(CAMERA_2_IP, queue2, "cam2"), daemon=True)
t1.start()
t2.start()

# -----------------------------
# Main loop
# -----------------------------
frame_count = 0
target_width, target_height = 400, 300

print("Real-time synchronized capture:")
print("Press 'c' to capture frame, 'q' to quit.")

while True:
    if queue1.empty() or queue2.empty():
        continue

    ts1, frame1 = queue1.get()
    ts2, frame2 = queue2.get()

    # Optional: enforce strict timestamp difference (e.g., <50ms)
    if abs(ts1 - ts2) > 0.05:  # adjust threshold if needed
        continue  # skip unsynced frames

    # MediaPipe processing
    rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    res1 = pose.process(rgb1)
    res2 = pose.process(rgb2)

    # Extract joints
    joints1 = np.full((33, 2), np.nan)
    joints2 = np.full((33, 2), np.nan)
    if res1.pose_landmarks:
        h, w, _ = frame1.shape
        for i, lm in enumerate(res1.pose_landmarks.landmark):
            joints1[i] = [lm.x * w, lm.y * h]
        mp_drawing.draw_landmarks(frame1, res1.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if res2.pose_landmarks:
        h, w, _ = frame2.shape
        for i, lm in enumerate(res2.pose_landmarks.landmark):
            joints2[i] = [lm.x * w, lm.y * h]
        mp_drawing.draw_landmarks(frame2, res2.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display resized for visualization
    vis1 = cv2.resize(frame1, (target_width, target_height))
    vis2 = cv2.resize(frame2, (target_width, target_height))
    combined = np.hstack((vis1, vis2))
    cv2.imshow("Cameras Side by Side (Skeleton)", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):
        # Save images
        cv2.imwrite(str(cam1_frame_dir / f"frame_{frame_count:03d}.png"), frame1)
        cv2.imwrite(str(cam2_frame_dir / f"frame_{frame_count:03d}.png"), frame2)
        # Save joints
        np.save(cam1_joint_dir / f"frame_{frame_count:03d}.npy", joints1)
        np.save(cam2_joint_dir / f"frame_{frame_count:03d}.npy", joints2)
        print(f"[Frame {frame_count}] Saved synchronized images + joints")
        frame_count += 1

    elif key == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cv2.destroyAllWindows()
pose.close()
print("Capture session ended.")
