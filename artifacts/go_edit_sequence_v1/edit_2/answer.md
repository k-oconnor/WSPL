This adds a complementary liberty certificate that subtracts possible diagonal double-counting, improving short-chain reasoning for both colors. Shape features are reconstructed later to free the necessary channels.

```wspl
wspl 0.2;
target base=artifact("parent",sha256="{{PARENT_SHA256}}");

const A=0;
const B=81;
const C=162;
const D=243;
const E=324;
const F=405;
const G=486;

edit base as candidate mode=fixed numeric=exact_to_storage {
  // Keep the original two-liberty seeds in h2 C/D.
  // Use h2 E/F for complementary enemy/friendly certificates.
  W2[:,E:G]=0;
  b2[E:G]=0;

  // Rebuild propagation, third-liberty evidence, and shape.
  W3[:,C:G]=0;
  b3[C:G]=0;

  for p in range(81) {
    // Complementary bound:
    // direct liberties + adjacent same-color direct liberties
    // minus every empty diagonal.
    //
    // Shared liberties of adjacent group members can occur only
    // at these diagonals. Subtracting all of them is conservative.
    W2[B+p,E+p]=12;
    W2[D+p,E+p]=1;
    b2[E+p]=-13;

    W2[A+p,F+p]=12;
    W2[C+p,F+p]=1;
    b2[F+p]=-13;

    // Merge both independent certificates and propagate only
    // through stones of the corresponding color.
    W3[A+p,C+p]=128;
    W3[C+p,C+p]=1;
    W3[F+p,C+p]=1;
    b3[C+p]=-128;

    W3[B+p,D+p]=128;
    W3[D+p,D+p]=1;
    W3[E+p,D+p]=1;
    b3[D+p]=-128;

    // Recover the original local third-liberty seed one layer
    // later: relu(relu(bound-1)-1) = relu(bound-2).
    W3[D+p,E+p]=1;
    b3[E+p]=-1;

    // Larger gating margins accommodate merged certificates.
    W4[A+p,C+p]=512;
    b4[C+p]=-512;
    W4[B+p,D+p]=512;
    b4[D+p]=-512;

    // Reconstruct the analytic spatial kernel directly from
    // the stone indicators still available in h2.
    b3[F+p]=parent.b1[F+p];
    for q in range(81) {
      W3[A+q,F+p]=parent.W1[3*q,F+p];
      W3[B+q,F+p]=parent.W1[3*q+1,F+p];
    }
  }

  // Horizontal directed adjacency.
  for r in range(9) {
    for c in range(8) {
      for s in range(2) {
        W2[D+9*r+c+1-s,E+9*r+c+s]+=1;
        W2[C+9*r+c+1-s,F+9*r+c+s]+=1;

        W3[C+9*r+c+1-s,C+9*r+c+s]+=1;
        W3[F+9*r+c+1-s,C+9*r+c+s]+=1;

        W3[D+9*r+c+1-s,D+9*r+c+s]+=1;
        W3[E+9*r+c+1-s,D+9*r+c+s]+=1;
      }
    }
  }

  // Vertical directed adjacency.
  for p in range(72) {
    for s in range(2) {
      W2[D+p+9*(1-s),E+p+9*s]+=1;
      W2[C+p+9*(1-s),F+p+9*s]+=1;

      W3[C+p+9*(1-s),C+p+9*s]+=1;
      W3[F+p+9*(1-s),C+p+9*s]+=1;

      W3[D+p+9*(1-s),D+p+9*s]+=1;
      W3[E+p+9*(1-s),D+p+9*s]+=1;
    }
  }

  // Subtract empty diagonals: occupancy minus one.
  // Reflections and reversal cover every directed diagonal.
  for r in range(8) {
    for c in range(8) {
      for d in range(2) {
        for s in range(2) {
          W2[A+9*r+c+d+(1-s)*(10-2*d),
             E+9*r+c+d+s*(10-2*d)]+=1;
          W2[B+9*r+c+d+(1-s)*(10-2*d),
             E+9*r+c+d+s*(10-2*d)]+=1;
          b2[E+9*r+c+d+s*(10-2*d)]-=1;

          W2[A+9*r+c+d+(1-s)*(10-2*d),
             F+9*r+c+d+s*(10-2*d)]+=1;
          W2[B+9*r+c+d+(1-s)*(10-2*d),
             F+9*r+c+d+s*(10-2*d)]+=1;
          b2[F+9*r+c+d+s*(10-2*d)]-=1;
        }
      }
    }
  }
}
```