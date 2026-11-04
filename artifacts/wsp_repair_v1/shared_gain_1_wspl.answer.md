Untie the first branch’s weight and set it to 3. The second branch retains weight 2; both zero biases remain unchanged.

```json
{
  "wspl": "wspl 0.2;\ntarget base=artifact(\"base\",sha256=\"{{base_sha256}}\");\nedit base as candidate mode=graph budget=\"repair\" {\n  untie parameter(\"n0.weight\") as W_drive;\n  W_drive[0,0]=3;\n}"
}
```