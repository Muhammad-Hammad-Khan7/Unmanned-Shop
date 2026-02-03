import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from .utils import load_joint_data, filter_valid_frames, estimate_fundamental

# -------------------------
# Paths
# -------------------------
DATA_FILE = "data/selected/joint_pairs.npy"
RESULT_FILE = "results/extrinsic_params.npy"
os.makedirs("results", exist_ok=True)

# -------------------------
# Load & filter data
# -------------------------
joint_data = load_joint_data(DATA_FILE)
valid_frames = filter_valid_frames(joint_data, conf_thresh=0.6, min_joints=6)

if len(valid_frames) < 20:
    raise RuntimeError("Not enough valid frames")

print(f"Using {len(valid_frames)} valid frames")

# -------------------------
# Image size (CHANGE if needed)
# -------------------------
IMG_W, IMG_H = 1280, 720
f = 0.9 * IMG_W  # weak perspective assumption

K = np.array([
    [f, 0, IMG_W / 2],
    [0, f, IMG_H / 2],
    [0, 0, 1]
], dtype=np.float64)

# -------------------------
# Estimate epipolar geometry
# -------------------------
F, pts1, pts2 = estimate_fundamental(valid_frames)
E = K.T @ F @ K

_, R_init, t_init, _ = cv2.recoverPose(E, pts1, pts2, K)

print("Initial pose estimated")

# -------------------------
# Optimization (NO SCALE)
# -------------------------
def reprojection_error(params, frames):
    rvec = params[:3]
    tdir = params[3:]
    tdir = tdir / np.linalg.norm(tdir)

    R, _ = cv2.Rodrigues(rvec)
    errors = []

    Kinv = np.linalg.inv(K)

    for _, joints in frames:
        for _, cam1, cam2 in joints:
            p1 = np.array([cam1[0], cam1[1], 1.0])
            p2 = np.array([cam2[0], cam2[1], 1.0])

            x1 = Kinv @ p1
            x2 = Kinv @ p2

            x1p = R @ x1 + tdir
            x1p /= x1p[2]

            errors.extend(x1p[:2] - x2[:2])

    return np.array(errors)

# Initial guess
rvec0, _ = cv2.Rodrigues(R_init)
x0 = np.hstack([rvec0.ravel(), t_init.ravel()])

res = least_squares(
    reprojection_error,
    x0,
    args=(valid_frames,),
    loss="cauchy",
    f_scale=1.0,
    verbose=2
)

# -------------------------
# Save results
# -------------------------
np.save(RESULT_FILE, res.x)

print("\nFINAL RESULTS")
print("Rotation vector:", res.x[:3])
print("Translation direction:", res.x[3:] / np.linalg.norm(res.x[3:]))
print("Final RMS Error:", np.sqrt(np.mean(res.fun ** 2)))
print(f"Saved to {RESULT_FILE}")
