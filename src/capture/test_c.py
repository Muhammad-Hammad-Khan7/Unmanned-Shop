import numpy as np

data = np.load(
    r"C:\Users\PMLS\Binocular_camera_calibration\data\selected\joint_pairs.npy",
    allow_pickle=True
)

print("Type:", type(data))
print("Length:", len(data))
print("Frame 0 type:", type(data[0]))
print("Frame 0 content:", data[0])
