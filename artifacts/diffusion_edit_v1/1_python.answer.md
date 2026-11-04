Copy-permute the pixel-input rows and output columns/biases, leaving time channels and hidden parameters unchanged. This implements \(g(z,t)=P f(P^{-1}z,t)\) without inference or training.

```python
import os
import numpy as np

names = ("W1", "b1", "W2", "b2", "W3", "b3")
with np.load("/parent.npz", allow_pickle=False) as parent:
    weights = {
        name: np.array(parent[name], dtype=np.float32, copy=True)
        for name in names
    }

# Each destination pixel receives the source two columns to its left.
source = np.roll(
    np.arange(144).reshape(12, 12), shift=2, axis=1
).ravel()

# Advanced indexing copies the RHS before assignment; time rows stay intact.
weights["W1"][:144, :] = weights["W1"][source, :]
weights["W3"] = weights["W3"][:, source]
weights["b3"] = weights["b3"][source]

os.makedirs("/output", exist_ok=True)
np.savez("/output/candidate.npz", **weights)
```