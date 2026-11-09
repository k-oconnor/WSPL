This refines liberty certificates using color-specific shared-liberty corrections, helping distinguish genuine atari from short groups whose liberties lie at different endpoints.

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
  // The positional kernel now lives in W5. Reuse its old
  // h1 channels for color-specific liberty-overlap bounds.
  W1[:,E:G]=0;
  b1[E:G]=0;

  // Replace only the complementary certificates.
  // Preserve the independent half-weighted certificates.
  W2[:,E:G]=0;
  b2[E:G]=0;

  for p in range(81) {
    // At an empty point:
    // E = relu(number of adjacent friendly stones - 1).
    // F = relu(number of adjacent opposing stones - 1).
    // Both vanish at occupied points.
    W1[3*p,E+p]=-4;
    W1[3*p+1,E+p]=-4;
    W1[3*p,F+p]=-4;
    W1[3*p+1,F+p]=-4;
    b1[E+p]=-1;
    b1[F+p]=-1;

    // Complementary two-liberty certificates:
    // direct liberties + neighboring same-color direct liberties
    // minus a conservative bound on duplicate liberties.
    W2[B+p,E+p]=12;
    W2[D+p,E+p]=1;
    b2[E+p]=-13;

    W2[A+p,F+p]=12;
    W2[C+p,F+p]=1;
    b2[F+p]=-13;
  }

  // Horizontal directed adjacency.
  for r in range(9) {
    for c in range(8) {
      for s in range(2) {
        W1[3*(9*r+c+1-s),E+9*r+c+s]+=1;
        W1[3*(9*r+c+1-s)+1,F+9*r+c+s]+=1;

        W2[D+9*r+c+1-s,E+9*r+c+s]+=1;
        W2[C+9*r+c+1-s,F+9*r+c+s]+=1;
      }
    }
  }

  // Vertical directed adjacency.
  for p in range(72) {
    for s in range(2) {
      W1[3*(p+9*(1-s)),E+p+9*s]+=1;
      W1[3*(p+9*(1-s))+1,F+p+9*s]+=1;

      W2[D+p+9*(1-s),E+p+9*s]+=1;
      W2[C+p+9*(1-s),F+p+9*s]+=1;
    }
  }

  // Neighboring group members can share liberties at diagonals.
  // Subtract contact redundancy rather than every empty diagonal:
  // an empty point touching only one same-color stone contributes
  // no duplicate-liberty correction.
  for r in range(8) {
    for c in range(8) {
      for d in range(2) {
        for s in range(2) {
          W2[F+9*r+c+d+(1-s)*(10-2*d),
             E+9*r+c+d+s*(10-2*d)]-=1;

          W2[E+9*r+c+d+(1-s)*(10-2*d),
             F+9*r+c+d+s*(10-2*d)]-=1;
        }
      }
    }
  }
}
```