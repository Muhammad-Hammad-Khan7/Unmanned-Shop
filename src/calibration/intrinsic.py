# src/calibration/intrinsic.py
import numpy as np
from scipy.optimize import least_squares
from .utils import load_joint_data, filter_valid_frames
import cv2
import os

# -------------------------
# Parameters
# -------------------------
DATA_FILE = "data/selected/joint_pairs.npy"  # your joint file
EXTRINSIC_FILE = "results/extrinsic_params.npy"
CONF_THRESH = 0.6
MIN_JOINTS = 6
RESULT_FILE = "results/intrinsic_params.npy"

os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)

# -------------------------
# Load data
# -------------------------
joint_data = load_joint_data(DATA_FILE)
valid_frames = filter_valid_frames(joint_data, CONF_THRESH, MIN_JOINTS)

if len(valid_frames) == 0:
    raise RuntimeError("No valid frames found for intrinsic optimization")

# Load extrinsic parameters
extrinsic = np.load(EXTRINSIC_FILE)
rvec = extrinsic[:3]
tvec = extrinsic[3:]

print(f"Loaded {len(valid_frames)} valid frames for intrinsic optimization.")
print(f"Using extrinsic parameters:\nRotation: {rvec}\nTranslation: {tvec}")

# -------------------------
# Triangulate 3D points from 2D correspondences
# -------------------------
def triangulate_points(cam1_pts, cam2_pts, rvec, tvec, fx, fy, cx, cy):
    """
    cam1_pts, cam2_pts: Nx2 arrays of 2D points
    rvec, tvec: extrinsic parameters
    fx, fy, cx, cy: intrinsics to optimize
    """
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,1)

    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    cam1_pts = cam1_pts.T
    cam2_pts = cam2_pts.T

    pts_4d = cv2.triangulatePoints(P1, P2, cam1_pts, cam2_pts)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d

# -------------------------
# Define reprojection error
# -------------------------
def reprojection_error_intrinsic(params, frames, rvec, tvec):
    fx, fy, cx, cy = params
    errors = []

    for _, joints in frames:
        # Collect corresponding points
        cam1_pts = []
        cam2_pts = []
        for jid, cam1, cam2 in joints:
            if cam1[0] > 0 and cam1[1] > 0 and cam2[0] > 0 and cam2[1] > 0:
                cam1_pts.append(cam1[:2])
                cam2_pts.append(cam2[:2])

        if len(cam1_pts) == 0:
            continue

        cam1_pts = np.array(cam1_pts)
        cam2_pts = np.array(cam2_pts)

        # Triangulate using current intrinsics
        pts_3d = triangulate_points(cam1_pts, cam2_pts, rvec, tvec, fx, fy, cx, cy)

        # Reproject to cam2
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3,1)
        for pt_3d, cam2_2d in zip(pts_3d, cam2_pts):
            pt_cam2 = R @ pt_3d.reshape(3,1) + t
            u = fx * pt_cam2[0,0] / pt_cam2[2,0] + cx
            v = fy * pt_cam2[1,0] / pt_cam2[2,0] + cy
            errors.extend([u - cam2_2d[0], v - cam2_2d[1]])

    return np.array(errors)

# -------------------------
# Run least squares
# -------------------------
# Initial guess: realistic for 1280x720 images
x0 = np.array([1000.0, 1000.0, 640.0, 360.0])

res = least_squares(
    reprojection_error_intrinsic,
    x0,
    args=(valid_frames, rvec, tvec),
    verbose=2
)

# -------------------------
# Save results
# -------------------------
np.save(RESULT_FILE, res.x)
print(f"Intrinsic parameters saved to {RESULT_FILE}")
print(f"Optimized intrinsic parameters:\nfx={res.x[0]}, fy={res.x[1]}, cx={res.x[2]}, cy={res.x[3]}")
