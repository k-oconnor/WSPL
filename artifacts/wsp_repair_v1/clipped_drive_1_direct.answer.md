Implement \( \mathrm{ReLU}(2x_0+\tfrac12)-\mathrm{ReLU}(2x_0-\tfrac12)-\tfrac12 \), which gives the requested saturation. Keep remains directly connected to \(x_1\).

```json
{
  "raw": {
    "put_tensors": {
      "clip_weight": {
        "shape": [1, 2],
        "values": [2, 2]
      },
      "clip_bias": {
        "shape": [2],
        "values": ["1/2", "-1/2"]
      },
      "drive_weight": {
        "shape": [2, 1],
        "values": [1, -1]
      },
      "drive_bias": {
        "shape": [1],
        "values": ["-1/2"]
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
          "weight": "clip_weight",
          "bias": "clip_bias"
        },
        "attrs": {
          "layout": "in_out"
        }
      },
      {
        "id": "clip_relu",
        "op": "relu",
        "inputs": {
          "x": "node:n0"
        },
        "params": {},
        "attrs": {}
      },
      {
        "id": "drive_clipped",
        "op": "affine",
        "inputs": {
          "x": "node:clip_relu"
        },
        "params": {
          "weight": "drive_weight",
          "bias": "drive_bias"
        },
        "attrs": {
          "layout": "in_out"
        }
      }
    ],
    "outputs": {
      "drive": "node:drive_clipped"
    }
  }
}
```