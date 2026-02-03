import numpy as np
# Load your current joint_pairs.npy
data = np.load("data/selected/joint_pairs.npy", allow_pickle=True)

fixed_data = []

for item in data:
    # If item is a list, assume it contains tuples
    if isinstance(item, list):
        fixed_data.append(item)
    # If item is a tuple with a list as second element
    elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], list):
        fixed_data.append(item[1])
    # If item is an int followed by a list
    elif isinstance(item, int):
        continue
    else:
        print("Skipping unexpected item:", item)

# Save the cleaned file
np.save("data/selected/joint_pairs_fixed.npy", fixed_data)
print("Saved fixed joint_pairs.npy with", len(fixed_data), "frames")
