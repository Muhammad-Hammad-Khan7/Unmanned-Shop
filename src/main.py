'''import cv2
import os
import numpy as np
from src.calibration.joint_optimization import joint_optimize
from src.pose.pose_extractor import extract_joints
from src.capture.sync_capture import capture_synced_frames, release_cameras
from src.capture.sync_capture import open_cameras

open_cameras()

def main():
    print("Binocular calibration pipeline started (Press SPACEBAR to capture frame, ESC or Q to exit)")

    # Load initial intrinsics (in pixels!)
    # Load the 640x480 matrices
    K1 = np.load("data/intrinsics/K_cam1.npy")
    K2 = np.load("data/intrinsics/K_cam2.npy")

    # Scale them for 1280x720
    # Width scale: 1280 / 640 = 2.0
    # Height scale: 720 / 480 = 1.5
    scale_w = 1280 / 640
    scale_h = 720 / 480

    for K in [K1, K2]:
        K[0, 0] *= scale_w  # Scale focal length x
        K[1, 1] *= scale_h  # Scale focal length y
        K[0, 2] *= scale_w  # Scale principal point x
        K[1, 2] *= scale_h  # Scale principal point y

    print("Adjusted K matrices for 1280x720 resolution")
    captured_pts1 = []
    captured_pts2 = []

    cam1_dir = "data/images/cam1"
    cam2_dir = "data/images/cam2"
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(cam2_dir, exist_ok=True)

    frame_count = 0

    while True:
        frame1, frame2 = capture_synced_frames()
        if frame1 is None or frame2 is None:
            print("Failed to capture frames, retrying...")
            continue

        # Display frames side by side safely
        # 1. Define a smaller preview size (Half of 1280x720)
        preview_w, preview_h = 640, 360

        # 2. Resize both frames to the preview size
        frame1_disp = cv2.resize(frame1, (preview_w, preview_h))
        frame2_disp = cv2.resize(frame2, (preview_w, preview_h))

        # 3. Stack them side-by-side
        combined_frame = np.hstack((frame1_disp, frame2_disp))

        # 4. Optional: Add a vertical line to separate the views
        cv2.line(combined_frame, (preview_w, 0), (preview_w, preview_h), (0, 255, 0), 2)

        # 5. Add text to show progress
        status_text = f"Captured: {frame_count}/15 | Press SPACE to save"
        cv2.putText(combined_frame, status_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Stereo Cameras - Left | Right", combined_frame)
        # --------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACEBAR → capture
            pts1, _ = extract_joints(frame1)
            pts2, _ = extract_joints(frame2)

            if pts1 is not None and pts2 is not None:
                # Convert normalized joints to pixels
                h1, w1 = frame1.shape[:2]
                h2, w2 = frame2.shape[:2]

                pts1_px = (pts1 * np.array([w1, h1])).astype(np.float32)
                pts2_px = (pts2 * np.array([w2, h2])).astype(np.float32)

                captured_pts1.append(pts1_px)
                captured_pts2.append(pts2_px)

                frame_count += 1

                # Save frames
                cv2.imwrite(os.path.join(cam1_dir, f"frame{frame_count}.jpg"), frame1)
                cv2.imwrite(os.path.join(cam2_dir, f"frame{frame_count}.jpg"), frame2)

                print(f"Captured frame {frame_count}")
            else:
                print("Pose not detected, try again.")

        elif key == 27 or key == ord('q'):  # ESC or Q → exit
            print("Manual exit")
            break

    cv2.destroyAllWindows()
    release_cameras()

    if len(captured_pts1) == 0:
        print("No frames captured. Exiting.")
        return

    # Run joint optimization
    print("Running joint optimization on captured frames...")
    K1, K2, R, t, points_3d, final_error = joint_optimize(
        captured_pts1,
        captured_pts2,
        K1,
        K2
    )

    print(f"Final average reprojection error: {final_error:.6f} pixels per joint")

    # Save calibration results
    intrinsics_dir = "data/intrinsics"
    extrinsics_dir = "data/extrinsics"
    os.makedirs(intrinsics_dir, exist_ok=True)
    os.makedirs(extrinsics_dir, exist_ok=True)

    np.save(os.path.join(intrinsics_dir, "K_cam1.npy"), K1)
    np.save(os.path.join(intrinsics_dir, "K_cam2.npy"), K2)
    np.save(os.path.join(extrinsics_dir, "R.npy"), R)
    np.save(os.path.join(extrinsics_dir, "t.npy"), t)

    print("Calibration complete. Parameters saved.")

if __name__ == "__main__":
    main()
'''
# main.py — Step 1: Initial Extrinsic Estimation
import numpy as np
from src.calibration.extrinsic import estimate_extrinsic  # Make sure this is correct

def main():
    print("=== Binocular Camera Calibration — Step 1 ===")

    # ------------------------------
    # Load 2D joint correspondences
    # ------------------------------
    captured_pts1 = np.load("data/joints/cam1_joints.npy", allow_pickle=True)
    captured_pts2 = np.load("data/joints/cam2_joints.npy", allow_pickle=True)

    # ------------------------------
    # Load intrinsic matrices
    # ------------------------------
    K1 = np.load("data/intrinsics/K_cam1.npy")
    K2 = np.load("data/intrinsics/K_cam2.npy")

    # ------------------------------
    # Step 1: Estimate initial extrinsics
    # ------------------------------
    try:
        R, t = estimate_extrinsic(captured_pts1, captured_pts2, K1, K2)
        print("\n[Step 1] Initial Extrinsics Estimated:")
        print("Rotation (R):\n", R)
        print("Translation (t):\n", t)
    except RuntimeError as e:
        print("Error during extrinsic estimation:", e)

if __name__ == "__main__":
    main()
