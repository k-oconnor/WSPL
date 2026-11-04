Give `drive` its own weight of 3, leaving `keep`’s weight of 2 and the shared zero bias unchanged. This satisfies both equalities within budget and FP32 tolerance.

```json
{
  "raw": {
    "put_tensors": {
      "W_drive": {
        "shape": [1, 1],
        "values": [3]
      }
    },
    "put_nodes": [
      {
        "id": "n0",
        "op": "affine",
        "inputs": {"x": "node:s0"},
        "params": {"weight": "W_drive", "bias": "b"},
        "attrs": {"layout": "in_out"}
      }
    ]
  }
}
```