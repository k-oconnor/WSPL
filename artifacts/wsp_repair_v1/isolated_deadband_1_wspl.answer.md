This computes the requested odd deadband using two ReLUs, removes cross-channel interference, and leaves the keep path unchanged. It uses 7 nodes and 8 stored parameters.

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
      "drive_weight": {
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
        "id": "n2",
        "op": "relu",
        "inputs": {"x": "node:n0"},
        "params": {},
        "attrs": {}
      },
      {
        "id": "drive_out",
        "op": "affine",
        "inputs": {"x": "node:n2"},
        "params": {
          "weight": "drive_weight",
          "bias": "b"
        },
        "attrs": {"layout": "in_out"}
      }
    ],
    "outputs": {
      "drive": "node:drive_out"
    }
  }
}
```