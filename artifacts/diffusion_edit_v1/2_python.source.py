import numpy as np
from pathlib import Path

names = ("W1", "b1", "W2", "b2", "W3", "b3")
destination = np.array(
    [12 * r + (c + 2) % 12 for r in range(12) for c in range(12)],
    dtype=np.intp,
)

with np.load("/parent.npz", allow_pickle=False) as parent:
    candidate = {
        name: np.array(parent[name], dtype=np.float32, copy=True)
        for name in names
    }

    candidate["W1"][destination, :] = parent["W1"][:144, :]
    candidate["W3"][:, destination] = parent["W3"]
    candidate["b3"][destination] = parent["b3"]

Path("/output").mkdir(parents=True, exist_ok=True)
np.savez("/output/candidate.npz", **candidate)
