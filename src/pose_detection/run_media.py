import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# -----------------------------
# Paths (MATCH YOUR SYSTEM)
# -----------------------------
ROOT = Path(r"C:\Users\PMLS\Binocular_camera_calibration")
FRAME_DIR = ROOT / "data" / "frames"
OUT_DIR = ROOT / "data" / "mediapipe"

CAMERAS = ["cam1", "cam2"]
NUM_JOINTS = 33

mp_pose = mp.solutions.pose

def process_camera(cam_name):
    img_dir = FRAME_DIR / cam_name
    save_dir = OUT_DIR / cam_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:

        for img_path in sorted(img_dir.glob("*")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                np.save(save_dir / f"{img_path.stem}.npy", None)
                continue

            joints = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

            for i, lm in enumerate(res.pose_landmarks.landmark):
                joints[i, 0] = lm.x * w
                joints[i, 1] = lm.y * h
                joints[i, 2] = lm.visibility   # confidence

            np.save(save_dir / f"{img_path.stem}.npy", joints)

            print(f"{cam_name} | {img_path.name} processed")

if __name__ == "__main__":
    for cam in CAMERAS:
        process_camera(cam)

    print("MediaPipe processing completed.")
