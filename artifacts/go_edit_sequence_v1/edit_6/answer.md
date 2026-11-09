This corrects false-eye handling: two opposing diagonals negate the eye penalty in the interior; one suffices on an edge or corner. Other features remain unchanged.

```wspl
wspl 0.2;
target base=artifact("parent",sha256="{{PARENT_SHA256}}");

const B=81;
const D=243;

edit base as candidate mode=fixed numeric=exact_to_storage {
  // Interior eyes tolerate one opposing diagonal, but not two.
  for r in range(8) {
    for c in range(8) {
      for d in range(2) {
        for s in range(2) {
          W5[B+9*r+c+d+(1-s)*(10-2*d),
             D+9*r+c+d+s*(10-2*d)]=-1;
        }
      }
    }
  }

  // Top and bottom edges: one opposing diagonal defeats an eye.
  for c in range(8) {
    W5[B+9+c+1,D+c]=-2;
    W5[B+9+c,D+c+1]=-2;

    W5[B+63+c+1,D+72+c]=-2;
    W5[B+63+c,D+72+c+1]=-2;
  }

  // Left and right edges, including corners.
  for r in range(8) {
    W5[B+9*(r+1)+1,D+9*r]=-2;
    W5[B+9*r+1,D+9*(r+1)]=-2;

    W5[B+9*(r+1)+7,D+9*r+8]=-2;
    W5[B+9*r+7,D+9*(r+1)+8]=-2;
  }
}
```