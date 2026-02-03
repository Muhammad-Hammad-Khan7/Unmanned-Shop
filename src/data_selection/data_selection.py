import numpy as np
from pathlib import Path

# -----------------------------
# Paths and parameters
# -----------------------------
ROOT = Path(r"C:\Users\PMLS\Binocular_camera_calibration")
MP_DIR = ROOT / "data" / "mediapipe"

CONF_THRESH = 0.6      # Visibility threshold (paper Step 3)
NUM_JOINTS = 33        # MediaPipe joints

# -----------------------------
# Get all frame files
# -----------------------------
cam1_files = sorted((MP_DIR / "cam1").glob("*.npy"))
cam2_files = sorted((MP_DIR / "cam2").glob("*.npy"))

if len(cam1_files) != len(cam2_files):
    raise ValueError(f"Frame count mismatch: {len(cam1_files)} vs {len(cam2_files)}")

# -----------------------------
# Data selection
# -----------------------------
selected = []

for i, (f1, f2) in enumerate(zip(cam1_files, cam2_files)):
    # Load MediaPipe output
    j1 = np.load(f1, allow_pickle=True)
    j2 = np.load(f2, allow_pickle=True)

    # -----------------------------
    # Skip invalid frames
    # -----------------------------
    if j1 is None or j2 is None:
        continue
    if not isinstance(j1, np.ndarray) or not isinstance(j2, np.ndarray):
        continue
    if j1.ndim != 2 or j2.ndim != 2:
        continue
    if j1.shape != (NUM_JOINTS, 3) or j2.shape != (NUM_JOINTS, 3):
        continue

    # -----------------------------
    # Select joints above confidence threshold
    # -----------------------------
    valid = []
    for j in range(NUM_JOINTS):
        if j1[j, 2] > CONF_THRESH and j2[j, 2] > CONF_THRESH:
            valid.append((j, j1[j], j2[j]))

    # Keep frame only if enough valid joints (>=8, paper assumption)
    if len(valid) >= 8:
        selected.append((i, valid))

# -----------------------------
# Ensure output directory exists
# -----------------------------
out_dir = ROOT / "data" / "selected"
out_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Save ragged list robustly
# -----------------------------
selected_array = np.array(selected, dtype=object)
np.save(out_dir / "joint_pairs.npy", selected_array, allow_pickle=True)

print(f"Selected {len(selected)} valid frames")
