Copy each pixel’s input row, output column, and output bias to its shifted position. Time inputs and hidden-layer parameters remain unchanged, preserving hidden activations and ensuring `g(P*x,t) = P*f(x,t)`.

```wspl
wspl 0.2;
target base=artifact("parent",sha256="{{PARENT_SHA256}}");

edit base as candidate mode=fixed numeric=exact_to_storage {
  for r in range(12) {
    for c in range(10) {
      W1[12*r+c+2,0:64] = parent.W1[12*r+c,0:64];
      W3[0:64,12*r+c+2] = parent.W3[0:64,12*r+c];
      b3[12*r+c+2] = parent.b3[12*r+c];
    }
    for c in range(10,12) {
      W1[12*r+c-10,0:64] = parent.W1[12*r+c,0:64];
      W3[0:64,12*r+c-10] = parent.W3[0:64,12*r+c];
      b3[12*r+c-10] = parent.b3[12*r+c];
    }
  }
}
```