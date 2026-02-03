# src/triangulate_3d.py

import numpy as np
import cv2

def triangulate_frame(joints, extrinsics, K):
    """
    Triangulate 3D points from two cameras.
    
    joints: list of (joint_id, cam1_xy, cam2_xy)
    extrinsics: 6 params: rvec(3) + tdir(3)
    K: 3x3 camera intrinsics
    """
    rvec = extrinsics[:3]
    tdir = extrinsics[3:]
    tdir = tdir / np.linalg.norm(tdir)  # normalize direction

    R, _ = cv2.Rodrigues(rvec)
    t = tdir.reshape(3,1)

    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t))

    pts1 = []
    pts2 = []
    ids = []

    for jid, cam1, cam2 in joints:
        if cam1[0] > 0 and cam1[1] > 0 and cam2[0] > 0 and cam2[1] > 0:  # only valid points
            pts1.append(cam1[:2])
            pts2.append(cam2[:2])
            ids.append(jid)

    if len(pts1) == 0:
        return {}

    pts1 = np.array(pts1).T  # 2 x N
    pts2 = np.array(pts2).T  # 2 x N

    pts_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T

    return dict(zip(ids, pts_3d))
