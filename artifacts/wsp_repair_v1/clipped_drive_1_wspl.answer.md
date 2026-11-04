Implement clipping as `ReLU(2x₀ + ½) − ReLU(2x₀ − ½) − ½`. Keep remains unchanged at `x₁`. This uses 7 nodes and 9 parameters, with FP32 rounding error safely below the tolerance.

```json
{
  "raw": {
    "put_tensors": {
      "clip_W": {
        "shape": [1, 2],
        "values": [1, 1]
      },
      "clip_b": {
        "shape": [2],
        "values": ["1/2", "-1/2"]
      },
      "combine_W": {
        "shape": [2, 1],
        "values": [1, -1]
      },
      "combine_b": {
        "shape": [1],
        "values": ["-1/2"]
      }
    },
    "put_nodes": [
      {
        "id": "clip_shift",
        "op": "affine",
        "inputs": {"x": "node:n0"},
        "params": {"weight": "clip_W", "bias": "clip_b"},
        "attrs": {"layout": "in_out"}
      },
      {
        "id": "clip_relu",
        "op": "relu",
        "inputs": {"x": "node:clip_shift"},
        "params": {},
        "attrs": {}
      },
      {
        "id": "clip_out",
        "op": "affine",
        "inputs": {"x": "node:clip_relu"},
        "params": {"weight": "combine_W", "bias": "combine_b"},
        "attrs": {"layout": "in_out"}
      }
    ],
    "outputs": {
      "drive": "node:clip_out"
    }
  }
}
```