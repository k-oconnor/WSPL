import numpy as np

W = [None] + [
    np.zeros(SHAPES[f"W{k}"], dtype=np.float32)
    for k in range(1, 8)
]
B = [None] + [
    np.zeros(SHAPES[f"b{k}"], dtype=np.float32)
    for k in range(1, 8)
]

def ch(channel, point):
    return 81 * channel + point

def add(layer, source, destination, value):
    W[layer][source, destination] += np.float32(value)

points = [(r, c) for r in range(9) for c in range(9)]
neighbors = []
diagonals = []
for r, c in points:
    neighbors.append([
        9 * rr + cc
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
        if 0 <= rr < 9 and 0 <= cc < 9
    ])
    diagonals.append([
        9 * (r+dr) + c+dc
        for dr in (-1, 1) for dc in (-1, 1)
        if 0 <= r+dr < 9 and 0 <= c+dc < 9
    ])

# Broad, reflection-symmetric local influence kernel.
density_weights = {
    (1, 0): 1.00,
    (1, 1): 0.65,
    (2, 0): 0.50,
    (2, 1): 0.32,
    (2, 2): 0.18,
    (3, 0): 0.22,
    (3, 1): 0.13,
    (3, 2): 0.07,
    (3, 3): 0.035,
    (4, 0): 0.08,
    (4, 1): 0.04,
}
density_offsets = []
for dr in range(-4, 5):
    for dc in range(-4, 5):
        key = tuple(sorted((abs(dr), abs(dc)), reverse=True))
        if key in density_weights:
            density_offsets.append((dr, dc, density_weights[key]))
full_density = sum(v for _, _, v in density_offsets)

# H1: own, opponent, own direct liberties, opponent direct liberties,
#     own enclosure, opponent enclosure.
for s, (r, c) in enumerate(points):
    add(1, 3*s, ch(0, s), 1)
    add(1, 3*s+1, ch(1, s), 1)

    for color, channel in ((0, 2), (1, 3)):
        dest = ch(channel, s)
        add(1, 3*s+color, dest, 4)
        B[1][dest] = len(neighbors[s]) - 4
        for t in neighbors[s]:
            add(1, 3*t, dest, -1)
            add(1, 3*t+1, dest, -1)

    local_density = 0.0
    for dr, dc, weight in density_offsets:
        rr, cc = r+dr, c+dc
        if not (0 <= rr < 9 and 0 <= cc < 9):
            continue
        t = 9*rr + cc
        local_density += weight
        add(1, 3*t, ch(4, s), weight)
        add(1, 3*t+1, ch(4, s), -0.85*weight)
        add(1, 3*t+1, ch(5, s), weight)
        add(1, 3*t, ch(5, s), -0.70*weight)

    threshold = 2.65 * (local_density / full_density)**0.70
    B[1][ch(4, s)] = -threshold
    B[1][ch(5, s)] = -threshold

# H2: own, opponent, three integer safety deficits, enclosure cost.
#
# For a stone i, P = 2*L(i) + sum_{same-color neighbors j} L(j).
# Every distinct liberty contributes at most twice.
# Thus P >= 3 proves at least two liberties, and P >= 5 proves three.
for s in range(81):
    for color in (0, 1):
        add(2, ch(color, s), ch(color, s), 1)

    for color, destination_channel, threshold in (
        (0, 2, 3),
        (1, 3, 3),
        (0, 4, 5),
    ):
        dest = ch(destination_channel, s)
        add(2, ch(color, s), dest, threshold)
        B[2][dest] = -2 * len(neighbors[s])
        for t in neighbors[s]:
            add(2, ch(0, t), dest, 2)
            add(2, ch(1, t), dest, 2)
            add(2, ch(2+color, t), dest, -1)

    add(2, ch(4, s), ch(5, s), 2.10)
    add(2, ch(5, s), ch(5, s), 2.85)

# H3: turn integer deficits into binary certified-safety indicators.
for s in range(81):
    for color in (0, 1):
        add(3, ch(color, s), ch(color, s), 1)
    for color, channel in ((0, 2), (1, 3), (0, 4)):
        add(3, ch(color, s), ch(channel, s), 1)
        add(3, ch(channel, s), ch(channel, s), -1)
    add(3, ch(5, s), ch(5, s), 1)

# H4: propagate safety across one same-color edge.
# Store its complement, restricted to stones of the appropriate color.
for s in range(81):
    for color in (0, 1):
        add(4, ch(color, s), ch(color, s), 1)
    for color, channel in ((0, 2), (1, 3), (0, 4)):
        dest = ch(channel, s)
        add(4, ch(color, s), dest, 1)
        add(4, ch(channel, s), dest, -1)
        for t in neighbors[s]:
            add(4, ch(channel, t), dest, -1)
    add(4, ch(5, s), ch(5, s), 1)

# H5: propagate certified safety one further same-color edge.
for s in range(81):
    for color in (0, 1):
        add(5, ch(color, s), ch(color, s), 1)
    for color, channel in ((0, 2), (1, 3), (0, 4)):
        dest = ch(channel, s)
        add(5, ch(channel, s), dest, 1)
        for t in neighbors[s]:
            add(5, ch(color, t), dest, -1)
            add(5, ch(channel, t), dest, 1)
    add(5, ch(5, s), ch(5, s), 1)

# H6: own, opponent, tactical stone urgency, move liberty deficit,
#     eye penalty, enclosure cost.
for s, (r, c) in enumerate(points):
    for color in (0, 1):
        add(6, ch(color, s), ch(color, s), 1)

    # Only uncertified stones produce tactical urgency.
    # A small neighboring-urgency term favors larger endangered groups.
    size_bonus = 0.35
    gate = 4 * size_bonus
    dest = ch(2, s)
    add(6, ch(2, s), dest, 7.5 + gate)
    add(6, ch(3, s), dest, 12.5 + gate)
    B[6][dest] = -gate
    for t in neighbors[s]:
        add(6, ch(2, t), dest, size_bonus)
        add(6, ch(3, t), dest, size_bonus)

    # Approximate post-move liberties:
    # adjacent empties, certified friendly support, possible captures.
    # Two-liberty support is discounted to mitigate duplicate liberties.
    dest = ch(3, s)
    B[6][dest] = 2 - len(neighbors[s])
    for t in neighbors[s]:
        add(6, ch(0, t), dest, -1)
        add(6, ch(1, t), dest, 1)
        add(6, ch(2, t), dest, 0.75)
        add(6, ch(4, t), dest, 1.25)
        add(6, ch(3, t), dest, -1.40)

    # Orthogonally surrounded friendly points; opposing diagonals
    # distinguish genuine eyes from likely false eyes.
    dest = ch(4, s)
    B[6][dest] = 1 - len(neighbors[s])
    for t in neighbors[s]:
        add(6, ch(0, t), dest, 1)
    diagonal_weight = 1.0 if r in (0, 8) or c in (0, 8) else 0.5
    for t in diagonals[s]:
        add(6, ch(1, t), dest, -diagonal_weight)

    add(6, ch(5, s), ch(5, s), 1)

# Global features occupy otherwise unused hidden units.
WHITE, OWN_COUNT, OPP_COUNT = 486, 487, 488
EARLY, MID, LATE = 489, 490, 491
CORNER_UNITS = list(range(492, 496))
GLOBAL_UNITS = list(range(486, 496))

add(1, 2, WHITE, 1)
B[1][EARLY] = 1
B[1][MID] = 1
B[1][LATE] = -48/20

for s in range(81):
    add(1, 3*s, OWN_COUNT, 1)
    add(1, 3*s+1, OPP_COUNT, 1)
    for color in (0, 1):
        add(1, 3*s+color, EARLY, -1/26)
        add(1, 3*s+color, MID, -1/52)
        add(1, 3*s+color, LATE, 1/20)

corner_specs = [
    (range(0, 4), range(0, 4), (2, 2)),
    (range(0, 4), range(5, 9), (2, 6)),
    (range(5, 9), range(0, 4), (6, 2)),
    (range(5, 9), range(5, 9), (6, 6)),
]
for unit, (rows, cols, center) in zip(CORNER_UNITS, corner_specs):
    B[1][unit] = 1
    for r in rows:
        for c in cols:
            s = 9*r+c
            add(1, 3*s, unit, -1)
            add(1, 3*s+1, unit, -1)

for layer in range(2, 7):
    for unit in GLOBAL_UNITS:
        add(layer, unit, unit, 1)

# Empty-corner encouragement vanishes as the opening ends.
for unit in CORNER_UNITS:
    add(2, OWN_COUNT, unit, -1/28)
    add(2, OPP_COUNT, unit, -1/28)

# Linear spatial shape preferences.
shape_weights = {
    (1, 0): (0.50, -0.45),
    (1, 1): (1.15, 0.10),
    (2, 0): (0.90, 0.45),
    (2, 1): (0.40, 0.20),
    (2, 2): (0.00, 0.05),
    (3, 0): (0.10, 0.10),
    (3, 1): (-0.05, 0.00),
}

for s, (r, c) in enumerate(points):
    for t in neighbors[s]:
        add(7, ch(2, t), s, 1)
    add(7, ch(3, s), s, -11.5)
    add(7, ch(4, s), s, -13.0)
    add(7, ch(5, s), s, -1)

    for t, (rr, cc) in enumerate(points):
        key = tuple(sorted((abs(rr-r), abs(cc-c)), reverse=True))
        if key in shape_weights:
            own_weight, opp_weight = shape_weights[key]
            add(7, ch(0, t), s, own_weight)
            add(7, ch(1, t), s, opp_weight)

    dr, dc = min(r, 8-r), min(c, 8-c)
    edge_distance = min(dr, dc)

    if dr == 0 and dc == 0:
        B[7][s] = -0.60
        add(7, MID, s, -0.80)
        opening = -3.0
    elif edge_distance == 0:
        B[7][s] = -0.35
        add(7, MID, s, -0.45)
        opening = -1.5
    elif edge_distance == 1:
        add(7, MID, s, -0.10)
        opening = 0.2 if dr == dc == 1 else 0.8
    elif dr == dc == 2:
        opening = 3.1
    elif edge_distance == 2:
        opening = 2.7
    elif r == c == 4:
        opening = 2.9
    else:
        opening = 2.5

    add(7, EARLY, s, opening)

    for unit, (_, _, (cr, cc)) in zip(CORNER_UNITS, corner_specs):
        distance_squared = (r-cr)**2 + (c-cc)**2
        bonus = 1.3 * np.exp(-0.5 * distance_squared)
        add(7, unit, s, bonus)

# Passing competes with unproductive territory filling, not normal play.
B[7][81] = -4.45
add(7, WHITE, 81, 0.90)
add(7, OWN_COUNT, 81, 0.035)
add(7, OPP_COUNT, 81, -0.035)
add(7, LATE, 81, 0.70)

# Explicit logit scaling, implemented entirely inside the fixed network.
W[7] *= np.float32(1.20)
B[7] *= np.float32(1.20)

assert sum(W[k].size + B[k].size for k in range(1, 8)) == 1_480_274
for k in range(1, 8):
    assert np.isfinite(W[k]).all()
    assert np.isfinite(B[k]).all()
    save(f"W{k}", W[k])
    save(f"b{k}", B[k])

print("Saved explicit Go heuristic circuits; no forward passes or fitting.")
