import numpy as np

W1, b1 = load("W1"), load("b1")
W2, b2 = load("W2"), load("b2")
W6, b6 = load("W6"), load("b6")

def ch(channel, point):
    return 81 * channel + point

neighbors = []
for r in range(9):
    for c in range(9):
        neighbors.append([
            9 * rr + cc
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if 0 <= rr < 9 and 0 <= cc < 9
        ])

# H1 channels 4 and 5 became unused after enclosure calculations moved
# to H6. Replace them with R(s) = max(L(s)-1, 0), gated by stone color.
# Existing H1 channels 2 and 3 already contain gated direct liberties L.
for s in range(81):
    for color in (0, 1):
        dest = ch(4 + color, s)
        W1[:, dest] = 0
        b1[dest] = len(neighbors[s]) - 5
        W1[3*s + color, dest] = 4
        for t in neighbors[s]:
            W1[3*t, dest] = -1
            W1[3*t + 1, dest] = -1

# Stronger, integer-valued local safety proofs.
#
# Two liberties:
#   F2 = 2*L(s) + sum_neighbors(L(t) + R(t)).
# A one-liberty group has F2 <= 2.
#
# Three liberties:
#   F3 = 4*L(s) - 2*R(s)
#        + sum_neighbors(L(t) + 2*R(t)).
# A group with at most two liberties has F3 <= 6:
# - L(s)=2: central contribution is 6, neighboring contributions zero.
# - L(s)=1: central contribution is 4, neighboring contributions <=2.
# - L(s)=0: neighboring L sum <=4 and neighboring R sum <=1.
# The last bound follows because two distinct neighbors of s cannot
# share two other orthogonal neighbors.
#
# These proofs include every certificate produced by the previous
# thresholds and additionally recognize useful neighboring-liberty cases.
# H3-H5 can retain their existing binary conversion and propagation.

for s in range(81):
    for color in (0, 1):
        stone = ch(color, s)
        direct = ch(2 + color, s)
        clipped = ch(4 + color, s)

        dest2 = ch(2 + color, s)
        W2[:, dest2] = 0
        b2[dest2] = 0
        W2[stone, dest2] = 3
        W2[direct, dest2] = -2
        for t in neighbors[s]:
            W2[ch(2 + color, t), dest2] = -1
            W2[ch(4 + color, t), dest2] = -1

        dest3 = ch(4 + color, s)
        W2[:, dest3] = 0
        b2[dest3] = 0
        W2[stone, dest3] = 7
        W2[direct, dest3] = -4
        W2[clipped, dest3] = 2
        for t in neighbors[s]:
            W2[ch(2 + color, t), dest3] = -1
            W2[ch(4 + color, t), dest3] = -2

# Preserve true-atari capture/rescue urgency while adding smaller
# defensive urgency for friendly stones lacking three-liberty proof.
size_bonus = 0.35
gate = 4 * size_bonus
defense_bonus = 1.50
attack_bonus = 2.60

for s in range(81):
    dest = ch(2, s)
    W6[:, dest] = 0
    b6[dest] = -gate

    W6[ch(2, s), dest] = 7.50 - defense_bonus
    W6[ch(4, s), dest] = defense_bonus + gate
    W6[ch(3, s), dest] = 12.50 - attack_bonus
    W6[ch(5, s), dest] = attack_bonus + gate

    for t in neighbors[s]:
        W6[ch(2, t), dest] += np.float32(size_bonus)
        W6[ch(3, t), dest] += np.float32(size_bonus)

for name, array in (
    ("W1", W1), ("b1", b1),
    ("W2", W2), ("b2", b2),
    ("W6", W6), ("b6", b6),
):
    assert np.isfinite(array).all()
    save(name, array)

print("Strengthened explicit liberty proofs and added defensive pressure.")
print("No forward passes, fitting, search, or gameplay queries.")
