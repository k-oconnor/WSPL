The drive computes `relu(x₀−¼)−relu(−x₀−¼)` independently of the second channel. The keep path remains unchanged. All added constants are exactly representable in FP32.

```json
{
  "raw": {
    "put_tensors": {
      "deadband_weight": {
        "shape": [1, 2],
        "values": [1, -1]
      },
      "deadband_bias": {
        "shape": [2],
        "values": ["-1/4", "-1/4"]
      },
      "difference_weight": {
        "shape": [2, 1],
        "values": [1, -1]
      }
    },
    "put_nodes": [
      {
        "id": "n0",
        "op": "affine",
        "inputs": {"x": "node:s0"},
        "params": {
          "weight": "deadband_weight",
          "bias": "deadband_bias"
        },
        "attrs": {"layout": "in_out"}
      },
      {
        "id": "deadband_relu",
        "op": "relu",
        "inputs": {"x": "node:n0"},
        "params": {},
        "attrs": {}
      },
      {
        "id": "n2",
        "op": "affine",
        "inputs": {"x": "node:deadband_relu"},
        "params": {
          "weight": "difference_weight",
          "bias": "b"
        },
        "attrs": {"layout": "in_out"}
      }
    ]
  }
}
```