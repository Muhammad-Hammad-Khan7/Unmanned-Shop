import numpy as np
import cv2

def load_joint_data(path):
    """
    Expected format:
    [
      (frame_idx, [
          (joint_id, cam1_xyc, cam2_xyc),
          ...
      ]),
      ...
    ]
    """
    return np.load(path, allow_pickle=True)

def filter_valid_frames(joint_data, conf_thresh=0.6, min_joints=6):
    valid = []
    for frame_idx, joints in joint_data:
        good = []
        for jid, cam1, cam2 in joints:
            if cam1[2] >= conf_thresh and cam2[2] >= conf_thresh:
                good.append((jid, cam1, cam2))
        if len(good) >= min_joints:
            valid.append((frame_idx, good))
    return valid

def estimate_fundamental(frames):
    pts1, pts2 = [], []

    for _, joints in frames:
        for _, cam1, cam2 in joints:
            pts1.append(cam1[:2])
            pts2.append(cam2[:2])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        cv2.FM_RANSAC,
        ransacReprojThreshold=2.0,
        confidence=0.999
    )

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    return F, pts1, pts2
