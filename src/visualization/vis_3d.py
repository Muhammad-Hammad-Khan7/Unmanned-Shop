import numpy as np
import cv2
from pathlib import Path

# =============================
# PATHS & PARAMETERS
# =============================
ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "frames"
SELECTED_FILE = ROOT / "data" / "selected" / "joint_pairs.npy"

WINDOW_HEIGHT = 720
WINDOW_WIDTH = 1280

GREEN = (0, 255, 0)
RED = (0, 0, 255)

SMOOTH_ALPHA = 0.75  # temporal smoothing strength

# =============================
# LOAD DATA
# =============================
selected = np.load(SELECTED_FILE, allow_pickle=True)

# =============================
# MEDIAPIPE BODY CONNECTIONS
# =============================
POSE_CONNECTIONS = [
    (11,12),
    (11,13),(13,15),
    (12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,31),
    (24,26),(26,28),(28,32)
]

# =============================
# IMAGE RESCALE (ASPECT SAFE)
# =============================
def rescale_image(img, target_h, target_w):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    px = (target_w - new_w) // 2
    py = (target_h - new_h) // 2

    canvas[py:py+new_h, px:px+new_w] = resized
    return canvas, px, py, scale

# =============================
# TORSO SIZE (ROBUST)
# =============================
def torso_size(joints):
    pairs = [(11,12), (23,24), (11,23), (12,24)]
    dists = []

    for a, b in pairs:
        if a in joints and b in joints:
            x1, y1, _ = joints[a]
            x2, y2, _ = joints[b]
            dists.append(np.hypot(x1-x2, y1-y2))

    return np.median(dists) if dists else 120

# =============================
# VALIDATE JOINT & BONE
# =============================
def valid_joint(x, y):
    return 0 <= x <= WINDOW_WIDTH and 0 <= y <= WINDOW_HEIGHT

def valid_bone(a, b, joints, torso):
    if a not in joints or b not in joints:
        return False

    x1, y1, _ = joints[a]
    x2, y2, _ = joints[b]

    dist = np.hypot(x1-x2, y1-y2)
    return 0.15 * torso < dist < 3.0 * torso

# =============================
# DRAW SKELETON
# =============================
def draw_skeleton(img, joints):
    torso = torso_size(joints)

    for a, b in POSE_CONNECTIONS:
        if valid_bone(a, b, joints, torso):
            x1, y1, _ = joints[a]
            x2, y2, _ = joints[b]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), GREEN, 2)

    for idx, (x, y, _) in joints.items():
        cv2.circle(img, (int(x), int(y)), 4, GREEN, -1)
        cv2.putText(img, str(idx), (int(x)+4, int(y)-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, RED, 1)

# =============================
# TEMPORAL MEMORY
# =============================
prev_cam1 = {}
prev_cam2 = {}

# =============================
# MAIN LOOP
# =============================
for frame_index, joints_list in selected[:100]:

    img1 = cv2.imread(str(IMG_DIR / "cam1" / f"frame_{frame_index:03d}.png"))
    img2 = cv2.imread(str(IMG_DIR / "cam2" / f"frame_{frame_index:03d}.png"))
    if img1 is None or img2 is None:
        continue

    canvas1, px1, py1, s1 = rescale_image(img1, WINDOW_HEIGHT, WINDOW_WIDTH)
    canvas2, px2, py2, s2 = rescale_image(img2, WINDOW_HEIGHT, WINDOW_WIDTH)

    cam1, cam2 = {}, {}

    for jid, c1, c2 in joints_list:
        x1, y1, z1 = c1
        x2, y2, z2 = c2

        x1p, y1p = x1*s1 + px1, y1*s1 + py1
        x2p, y2p = x2*s2 + px2, y2*s2 + py2

        if valid_joint(x1p, y1p):
            if jid in prev_cam1:
                x1p = SMOOTH_ALPHA*prev_cam1[jid][0] + (1-SMOOTH_ALPHA)*x1p
                y1p = SMOOTH_ALPHA*prev_cam1[jid][1] + (1-SMOOTH_ALPHA)*y1p
            cam1[jid] = [x1p, y1p, z1]
            prev_cam1[jid] = cam1[jid]

        if valid_joint(x2p, y2p):
            if jid in prev_cam2:
                x2p = SMOOTH_ALPHA*prev_cam2[jid][0] + (1-SMOOTH_ALPHA)*x2p
                y2p = SMOOTH_ALPHA*prev_cam2[jid][1] + (1-SMOOTH_ALPHA)*y2p
            cam2[jid] = [x2p, y2p, z2]
            prev_cam2[jid] = cam2[jid]

    draw_skeleton(canvas1, cam1)
    draw_skeleton(canvas2, cam2)

    cv2.imshow("Stable 2D Skeleton (No Collapse)", np.hstack([canvas1, canvas2]))
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
