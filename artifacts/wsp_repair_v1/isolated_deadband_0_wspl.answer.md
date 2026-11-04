The new drive computes `ReLU(x₀−¼)−ReLU(−x₀−¼)`, removing cross-channel interference. The keep branch and its parameters remain unchanged.

```json
{
  "raw": {
    "drop_nodes": ["n0", "n2"],
    "put_tensors": {
      "deadband_weight": {
        "shape": [1, 2],
        "values": [1, -1]
      },
      "deadband_bias": {
        "shape": [2],
        "values": ["-1/4", "-1/4"]
      },
      "drive_weight": {
        "shape": [2, 1],
        "values": [1, -1]
      }
    },
    "put_nodes": [
      {
        "id": "deadband_offsets",
        "op": "affine",
        "inputs": {"x": "node:s0"},
        "params": {
          "weight": "deadband_weight",
          "bias": "deadband_bias"
        },
        "attrs": {"layout": "in_out"}
      },
      {
        "id": "deadband_rectifiers",
        "op": "relu",
        "inputs": {"x": "node:deadband_offsets"},
        "params": {},
        "attrs": {}
      },
      {
        "id": "repaired_drive",
        "op": "affine",
        "inputs": {"x": "node:deadband_rectifiers"},
        "params": {
          "weight": "drive_weight",
          "bias": "b"
        },
        "attrs": {"layout": "in_out"}
      }
    ],
    "outputs": {
      "drive": "node:repaired_drive"
    }
  }
}
```