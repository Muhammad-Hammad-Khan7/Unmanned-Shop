import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Simple skeleton connections (MediaPipe-like)
SKELETON = [
(11, 12), # shoulders
(11, 23), (12, 24), # torso
(23, 24),
]




def plot_skeleton(joints_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for (i, j) in SKELETON:
        if i in joints_3d and j in joints_3d:
            p1, p2 = joints_3d[i], joints_3d[j]
            ax.plot([p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]], 'b')


    for p in joints_3d.values():
        ax.scatter(p[0], p[1], p[2], c='r')
        ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Human Skeleton')
    plt.show()