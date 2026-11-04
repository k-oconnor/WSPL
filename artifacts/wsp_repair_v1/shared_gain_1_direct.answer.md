Assign a new weight of 3 only to the drive branch. The keep branch retains weight 2, and both retain zero bias.

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
        "inputs": {
          "x": "node:s0"
        },
        "params": {
          "weight": "W_drive",
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