The drive now computes `relu(x[0]−1/4)−relu(−x[0]−1/4)`, eliminating cross-channel interference. The keep branch remains unchanged. All constants are exact in FP32, and the repair stays within budget.

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
        "inputs": {
          "x": "node:s0"
        },
        "params": {
          "weight": "deadband_weight",
          "bias": "deadband_bias"
        },
        "attrs": {
          "layout": "in_out"
        }
      },
      {
        "id": "deadband_relu",
        "op": "relu",
        "inputs": {
          "x": "node:n0"
        },
        "params": {},
        "attrs": {}
      },
      {
        "id": "n2",
        "op": "affine",
        "inputs": {
          "x": "node:deadband_relu"
        },
        "params": {
          "weight": "drive_weight",
          "bias": "b"
        },
        "attrs": {
          "layout": "in_out"
        }
      }
    ]
  }
}
```