import numpy as np

# Coefficient construction only; no positions or network evaluations.
W2 = load("W2")
b2 = load("b2")

def u(point, channel):
    return 6 * point + channel

corners = {0, 8, 72, 80}
changed = 0

for point in range(81):
    for color in (0, 1):
        # Friendly corner seeds include a distinct-liberty correction
        # whose existing summed-threshold calibration must be retained.
        if color == 0 and point in corners:
            continue

        for axis in (0, 1):
            destination = u(point, 2 + 2*axis + color)

            assert b2[destination] == np.float32(-17)
            assert W2[u(point, color), destination] == np.float32(16)

            # For a stone of this color, the existing preactivation
            # is L-1, where L is an integer liberty lower bound.
            #
            # Replace relu(L-1) with relu(2*L-3):
            #   L <= 1: 0
            #   L == 2: 1
            #   L >= 3: at least 3
            #
            # The existing h3 test relu(horizontal + vertical - 2)
            # now detects either directional three-liberty proof,
            # without confusing two separate two-liberty proofs.
            # The positive support of each two-liberty seed is unchanged.
            W2[:, destination] *= np.float32(2)
            b2[destination] = (
                np.float32(2) * b2[destination] - np.float32(1)
            )
            changed += 1

assert changed == 316

# Conservative symbolic magnitude bounds remain below existing gates:
# directional seed <= 13; supplemental three-liberty seed <= 4.
# h3 two-proof sum <= 5*(2*13)+2 = 132 < 256.
# h3 three-proof sum <= 2*13 + 5*6*4 = 146 < 256.
# h4 incoming sums, including auxiliary proofs, remain below 2048.

save("W2", W2)
save("b2", b2)

print("Sharpened 316 directional liberty seeds using integer thresholds.")
print("Preserved calibrated corner circuits and all scoring coefficients.")
