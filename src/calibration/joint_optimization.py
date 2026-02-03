# src/calibration/joint_optimization.py
import numpy as np
from scipy.optimize import least_squares
from .utils import load_joint_data, filter_valid_frames, rodrigues_to_matrix

# -------------------------
# File paths
# -------------------------
JOINT_FILE = "data/selected/joint_pairs.npy"
EXTRINSIC_FILE = "results/extrinsic_params.npy"
INTRINSIC_FILE = "results/intrinsic_params.npy"
OUTPUT_FILE = "results/joint_optimized_params.npy"

# -------------------------
# Parameters
# -------------------------
CONF_THRESH = 0.6
MIN_JOINTS = 6

# -------------------------
# Load joint data
# -------------------------
joint_data = load_joint_data(JOINT_FILE)
valid_frames = filter_valid_frames(joint_data, CONF_THRESH, MIN_JOINTS)
if len(valid_frames) == 0:
    raise RuntimeError("No valid frames found for optimization")
print(f"Loaded {len(valid_frames)} valid frames for joint optimization.")

# -------------------------
# Load previous calibration
# -------------------------
extrinsic = np.load(EXTRINSIC_FILE, allow_pickle=True)
rvec_init = extrinsic[:3]
tvec_init = extrinsic[3:]

intrinsic = np.load(INTRINSIC_FILE, allow_pickle=True)
fx_init, fy_init, cx_init, cy_init = intrinsic

print("Using initial extrinsic parameters:")
print("Rotation vector:", rvec_init)
print("Translation vector:", tvec_init)
print("Using initial intrinsic parameters:")
print("fx, fy, cx, cy =", fx_init, fy_init, cx_init, cy_init)

# -------------------------
# Define reprojection error
# -------------------------
def reprojection_error_joint(params, frames):
    """
    params: [rvec(3), tvec(3), fx, fy, cx, cy]
    frames: list of (frame_index, list_of_joints)
    """
    rvec = params[:3]
    tvec = params[3:6]
    fx, fy, cx, cy = params[6:]

    R = rodrigues_to_matrix(rvec)
    errors = []

    for frame_index, joints in frames:
        for jid, cam1, cam2 in joints:
            # Transform cam1 3D joint using extrinsic (R, t)
            joint_cam1_3d = np.array([cam1[0], cam1[1], 1.0])  # placeholder z=1
            joint_world = R @ joint_cam1_3d + tvec

            # Project to 2D using intrinsic parameters
            u_proj = fx * joint_world[0] / joint_world[2] + cx
            v_proj = fy * joint_world[1] / joint_world[2] + cy

            # Compare with cam2 2D
            err_x = u_proj - cam2[0]
            err_y = v_proj - cam2[1]
            errors.extend([err_x, err_y])

    return np.array(errors)

# -------------------------
# Run joint optimization
# -------------------------
x0 = np.hstack([rvec_init, tvec_init, fx_init, fy_init, cx_init, cy_init])
res = least_squares(
    reprojection_error_joint,
    x0,
    args=(valid_frames,),
    verbose=2,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
)

# -------------------------
# Save optimized parameters
# -------------------------
optimized_params = res.x
np.save(OUTPUT_FILE, optimized_params)
print(f"Joint optimized parameters saved to {OUTPUT_FILE}")

# -------------------------
# Display results
# -------------------------
print("Final optimized parameters:")
print("Rotation vector:", optimized_params[:3])
print("Translation vector:", optimized_params[3:6])
print("Intrinsic parameters: fx, fy, cx, cy =", optimized_params[6:])
