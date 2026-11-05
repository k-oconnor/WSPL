import numpy as np

W5 = load("W5")
b5 = load("b5")
W7 = load("W7")

def u(channel, point):
    return 81 * channel + point

gain = 1.30

for r in range(9):
    for c in range(9):
        q = 9 * r + c
        neighbors = [
            9 * rr + cc
            for rr, cc in (
                (r - 1, c), (r + 1, c),
                (r, c - 1), (r, c + 1)
            )
            if 0 <= rr < 9 and 0 <= cc < 9
        ]
        diagonals = [
            9 * rr + cc
            for rr, cc in (
                (r - 1, c - 1), (r - 1, c + 1),
                (r + 1, c - 1), (r + 1, c + 1)
            )
            if 0 <= rr < 9 and 0 <= cc < 9
        ]

        score = u(0, q)
        eye = u(4, q)
        contact = u(5, q)

        for p in neighbors:
            # Broaden attack preference below the capture-like threshold.
            # Above that threshold, the combined slope remains:
            # old: 1.6 + 2.6; new: 3.0 + 1.2.
            W5[u(3, p), score] = np.float32(3.0)

            # Modest preventive defense, separate from urgent rescue.
            # Existing self-atari and eye penalties remain in force.
            W5[u(2, p), score] = np.float32(0.45)

            # Contact with a vulnerable enemy is less objectionable than
            # contact with a healthy enemy. Capture-based relief remains.
            W5[u(3, p), contact] = np.float32(-0.70)

        W7[u(1, q), q] = np.float32(gain * 1.8)
        W7[u(2, q), q] = np.float32(gain * 1.2)

        # Interior true eyes tolerate one hostile diagonal; boundary eyes
        # tolerate none. Missing any friendly orthogonal neighbor turns
        # this feature off.
        degree = len(neighbors)
        if degree == 4:
            b5[eye] = np.float32(2.0 - 2.0 * degree)
            diagonal_penalty = -1.0
        else:
            b5[eye] = np.float32(1.5 - 2.0 * degree)
            diagonal_penalty = -1.5

        for p in diagonals:
            W5[u(1, p), eye] = np.float32(diagonal_penalty)

save("W5", W5)
save("b5", b5)
save("W7", W7)

print("Refined attack, preventive-defense, contact, and true-eye circuits.")
print("No training, search, simulations, or neural evaluations performed.")
