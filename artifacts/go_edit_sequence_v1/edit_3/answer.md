This adds conservative three-liberty support for connecting moves. It also removes the self-atari penalty for isolated, fully surrounded captures, while retaining it for potential snapbacks involving friendly groups.

```wspl
wspl 0.2;
target base=artifact("parent",sha256="{{PARENT_SHA256}}");

const A=0;
const B=81;
const C=162;
const E=324;
const F=405;
const G=486;

edit base as candidate mode=fixed numeric=exact_to_storage {
  // Move the positional kernel to h5, freeing h3/h4 F
  // for friendly three-liberty certificates.
  W3[:,F:G]=0;
  b3[F:G]=0;
  W4[:,F:G]=0;
  b4[F:G]=0;
  W5[:,F:G]=0;
  b5[F:G]=0;

  // Enemy pressure is nonnegative before its existing ReLU.
  // Fold it into the shifted score, freeing h6 E.
  W6[:,E:F]=0;
  b6[E:F]=0;

  for p in range(81) {
    // The original friendly seed is relu(liberty_bound-1).
    // Thresholding it again certifies at least three liberties.
    W3[C+p,F+p]=1;
    b3[F+p]=-1;

    // Propagate this certificate through friendly stones.
    W4[A+p,F+p]=128;
    W4[F+p,F+p]=1;
    b4[F+p]=-128;

    // Reconstruct the unchanged positional kernel from the
    // exact stone indicators retained through h4.
    b5[F+p]=parent.b3[F+p];
    for q in range(81) {
      W5[A+q,F+p]=parent.W3[A+q,F+p];
      W5[B+q,F+p]=parent.W3[B+q,F+p];
    }

    // h5 C is at most six, reaching six only when every
    // orthogonal neighbor is opposing.
    //
    // relu(2*h5C - 10 - capture_contacts) is zero elsewhere.
    // At a surrounded legal move, it equals the existing
    // liberty deficit: legality guarantees an actual capture.
    W6[C+p,E+p]=2;
    b6[E+p]=-10;
    W7[E+p,p]=11;
  }

  // Horizontal directed adjacency.
  for r in range(9) {
    for c in range(8) {
      for s in range(2) {
        W4[F+9*r+c+1-s,F+9*r+c+s]+=1;

        // A neighboring three-liberty group guarantees at
        // least two liberties remain after connecting to it.
        W5[F+9*r+c+1-s,C+9*r+c+s]-=8;

        // Preserve the former pressure contribution exactly
        // as an affine contribution to the shifted score.
        W6[E+9*r+c+1-s,F+9*r+c+s]+=12/5;
        W6[B+9*r+c+1-s,F+9*r+c+s]-=12/5;

        W6[B+9*r+c+1-s,E+9*r+c+s]-=1;
      }
    }
  }

  // Vertical directed adjacency.
  for p in range(72) {
    for s in range(2) {
      W4[F+p+9*(1-s),F+p+9*s]+=1;

      W5[F+p+9*(1-s),C+p+9*s]-=8;

      W6[E+p+9*(1-s),F+p+9*s]+=12/5;
      W6[B+p+9*(1-s),F+p+9*s]-=12/5;

      W6[B+p+9*(1-s),E+p+9*s]-=1;
    }
  }
}
```