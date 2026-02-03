import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.triangulate_3d import triangulate_frame

# Load data
joint_data = np.load('data/selected/joint_pairs.npy', allow_pickle=True)
extrinsics = np.load('results/extrinsic_params.npy')

# Skeleton connections (MediaPipe-like)
SKELETON = [
    (11, 12),  # shoulders
    (11, 23), (12, 24),  # torso
    (23, 24),
]

# Approximate camera intrinsics (for triangulation)
IMG_W, IMG_H = 1280, 720
f = 0.9 * IMG_W
K = np.array([[f, 0, IMG_W/2],
              [0, f, IMG_H/2],
              [0, 0, 1]], dtype=np.float64)

# Compute global bounds for axes scaling
all_points = []
for _, joints in joint_data:
    pts = triangulate_frame(joints, extrinsics, K)
    if pts:
        all_points.append(np.array(list(pts.values())))
if all_points:
    all_points = np.vstack(all_points)
    x_min, y_min, z_min = all_points.min(axis=0)
    x_max, y_max, z_max = all_points.max(axis=0)
else:
    x_min=y_min=z_min=-1
    x_max=y_max=z_max=1

# Frame index
frame_idx = [0]
num_frames = len(joint_data)

# Create 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def plot_frame(idx):
    ax.cla()
    ax.set_xlim([x_min-0.5, x_max+0.5])
    ax.set_ylim([y_min-0.5, y_max+0.5])
    ax.set_zlim([z_min-0.5, z_max+0.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Frame {idx}')

    _, joints = joint_data[idx]
    joints_3d = triangulate_frame(joints, extrinsics, K)

    # Draw bones
    for i, j in SKELETON:
        if i in joints_3d and j in joints_3d:
            p1, p2 = joints_3d[i], joints_3d[j]
            ax.plot([p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]], 'b', linewidth=2)

    # Draw joints
    for p in joints_3d.values():
        ax.scatter(p[0], p[1], p[2], c='r', s=40)

    plt.draw()
    plt.pause(0.001)

def on_key(event):
    # Press Right Arrow or Enter → next frame
    if event.key in ['right', 'enter']:
        frame_idx[0] = (frame_idx[0]+1) % num_frames
        plot_frame(frame_idx[0])
    # Press Left Arrow → previous frame
    elif event.key in ['left']:
        frame_idx[0] = (frame_idx[0]-1) % num_frames
        plot_frame(frame_idx[0])
    # Press Esc or q → quit
    elif event.key in ['escape','q']:
        plt.close(fig)

# Connect key events
fig.canvas.mpl_connect('key_press_event', on_key)

# Plot first frame
plot_frame(frame_idx[0])
plt.show()
