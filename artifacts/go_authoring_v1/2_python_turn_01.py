import numpy as np

names = ("b3", "W5", "b5", "W6", "b6", "W7", "b7")
a = {name: load(name) for name in names}

def u(channel, point):
    return 81 * channel + point

orth = []
for r in range(9):
    for c in range(9):
        orth.append([
            9 * rr + cc
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if 0 <= rr < 9 and 0 <= cc < 9
        ])

gain = 1.30

for q in range(81):
    # Preserve the connected-safety circuit, but retain more evidence
    # of groups with two liberties rather than only near-atari groups.
    a["b3"][u(2, q)] = -5.0
    a["b3"][u(3, q)] = -5.0

    # Threshold rescue and capture-like pressure separately.
    a["b5"][u(1, q)] = -0.5
    a["b5"][u(2, q)] = -0.8

    # These tactical candidate features must vanish at occupied points.
    # This also makes their later global aggregates meaningful.
    for color in (0, 1):
        a["W5"][u(color, q), u(1, q)] = -32.0
        a["W5"][u(color, q), u(2, q)] = -32.0

    for p in orth[q]:
        # Raw risk:
        # relu(2 - empty_neighbors - 2*friendly_neighbors
        #      + sum(friendly_vulnerability)).
        # A weak friendly neighbor therefore supplies less escape credit.
        a["W5"][u(2, p), u(3, q)] = 1.0

        # Ordinary liberty pressure contributes to strategic preference,
        # without automatically granting capture-related risk relief.
        a["W5"][u(3, p), u(0, q)] += np.float32(1.6)

    # Only the thresholded capture-like feature relieves these penalties.
    a["W6"][u(2, q), u(3, q)] = -1.8
    a["W6"][u(2, q), u(5, q)] = -1.3

    a["W7"][u(2, q), q] = gain * 2.6
    a["W7"][u(3, q), q] = -gain * 4.2

# Previously unused neurons summarize empty-point tactical opportunities.
CAPTURE_ACTIVITY = 495
RESCUE_ACTIVITY = 496

for target in (CAPTURE_ACTIVITY, RESCUE_ACTIVITY):
    a["W6"][:, target] = 0.0
    a["b6"][target] = 0.0
    a["W7"][target, :] = 0.0

for q in range(81):
    a["W6"][u(2, q), CAPTURE_ACTIVITY] = 1.0
    a["W6"][u(1, q), RESCUE_ACTIVITY] = 1.0

# Retain the existing opening suppression and bounded lead adjustments.
# Passing remains available when placements are strategically unattractive.
a["b7"][PASS] = -2.0 * gain
a["W7"][CAPTURE_ACTIVITY, PASS] = -0.14 * gain
a["W7"][RESCUE_ACTIVITY, PASS] = -0.08 * gain

for name in names:
    save(name, a[name])

print("Updated tactical thresholds, escape-credit calibration, and passing.")
print("Construction used coefficient edits only; no policy evaluations.")
