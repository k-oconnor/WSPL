Use `ReLU(2x₀+½) − ReLU(2x₀−½) − ½` to saturate drive. Keep remains directly connected to `x₁`; all additions fit the budget.

```json
{
  "raw": {
    "put_tensors": {
      "clip_shift_W": {
        "shape": [1, 2],
        "values": [1, 1]
      },
      "clip_shift_b": {
        "shape": [2],
        "values": ["1/2", "-1/2"]
      },
      "clip_out_W": {
        "shape": [2, 1],
        "values": [1, -1]
      },
      "clip_out_b": {
        "shape": [1],
        "values": ["-1/2"]
      }
    },
    "put_nodes": [
      {
        "id": "clip_shift",
        "op": "affine",
        "inputs": {"x": "node:n0"},
        "params": {
          "weight": "clip_shift_W",
          "bias": "clip_shift_b"
        },
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
        "params": {
          "weight": "clip_out_W",
          "bias": "clip_out_b"
        },
        "attrs": {"layout": "in_out"}
      }
    ],
    "outputs": {
      "drive": "node:clip_out"
    }
  }
}
```