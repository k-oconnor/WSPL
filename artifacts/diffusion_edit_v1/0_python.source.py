import os
import numpy as np

names = ("W1", "b1", "W2", "b2", "W3", "b3")
with np.load("/parent.npz", allow_pickle=False) as parent:
    tensors = {
        name: np.array(parent[name], dtype=np.float32, copy=True)
        for name in names
    }

# For each destination pixel, identify its source under the rightward shift.
source = np.roll(
    np.arange(144).reshape(12, 12), shift=2, axis=1
).ravel()

# Advanced indexing copies the source, so this assignment is overlap-safe.
tensors["W1"][:144, :] = tensors["W1"][source, :]
tensors["W3"] = tensors["W3"][:, source].copy()
tensors["b3"] = tensors["b3"][source].copy()

os.makedirs("/output", exist_ok=True)
np.savez("/output/candidate.npz", **tensors)
