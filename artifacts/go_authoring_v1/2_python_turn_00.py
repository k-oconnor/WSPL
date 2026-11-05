import numpy as np

# Construct every coefficient explicitly from geometry and Go heuristics.
W = [np.zeros(SHAPES[f"W{k}"], dtype=np.float32) for k in range(1, 8)]
B = [np.zeros(SHAPES[f"b{k}"], dtype=np.float32) for k in range(1, 8)]

N = 81
coords = [(r, c) for r in range(9) for c in range(9)]
orth = []
diag = []
for r, c in coords:
    orth.append([
        9 * rr + cc
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
        if 0 <= rr < 9 and 0 <= cc < 9
    ])
    diag.append([
        9 * rr + cc
        for rr, cc in ((r-1, c-1), (r-1, c+1),
                       (r+1, c-1), (r+1, c+1))
        if 0 <= rr < 9 and 0 <= cc < 9
    ])

def unit(channel, point):
    return 81 * channel + point

def edge(layer, source, target, weight):
    W[layer-1][source, target] += np.float32(weight)

def bias(layer, target, value):
    B[layer-1][target] = np.float32(value)

def copy_channel(layer, source_channel, target_channel):
    for q in range(N):
        edge(layer, unit(source_channel, q), unit(target_channel, q), 1.0)

# Global nonnegative signals, retained independently of the spatial channels.
OWN_COUNT = 486
OPP_COUNT = 487
WHITE = 488
OPENING = 489
EARLY_PASS = 490
GLOBALS = [OWN_COUNT, OPP_COUNT, WHITE, OPENING, EARLY_PASS]

# ---------------------------------------------------------------------------
# h1:
# 0 own occupancy, 1 opponent occupancy
# 2/3 individual stone liberties, gated by color
# 4/5 individual liberty surplus above one, gated by color
# ---------------------------------------------------------------------------
for q in range(N):
    degree = len(orth[q])
    for color in (0, 1):
        edge(1, 3*q + color, unit(color, q), 1.0)

        liberties = unit(2 + color, q)
        surplus = unit(4 + color, q)
        bias(1, liberties, degree - 8.0)
        bias(1, surplus, degree - 9.0)
        edge(1, 3*q + color, liberties, 8.0)
        edge(1, 3*q + color, surplus, 8.0)
        for p in orth[q]:
            for occupied_color in (0, 1):
                edge(1, 3*p + occupied_color, liberties, -1.0)
                edge(1, 3*p + occupied_color, surplus, -1.0)

    edge(1, 3*q, OWN_COUNT, 1.0)
    edge(1, 3*q + 1, OPP_COUNT, 1.0)
    for color in (0, 1):
        edge(1, 3*q + color, OPENING, -1.0 / 55.0)
        edge(1, 3*q + color, EARLY_PASS, -1.0)

edge(1, 2, WHITE, 1.0)
bias(1, OPENING, 1.0)
bias(1, EARLY_PASS, 45.0)

# ---------------------------------------------------------------------------
# h2: same-color one-edge propagation.
# 2/3 collect liberties from the stone and its immediate connected neighbors.
# 4/5 propagate evidence of a stone having at least two direct liberties.
# ---------------------------------------------------------------------------
copy_channel(2, 0, 0)
copy_channel(2, 1, 1)
for q in range(N):
    for color in (0, 1):
        for channel in (2 + color, 4 + color):
            target = unit(channel, q)
            bias(2, target, -32.0)
            edge(2, unit(color, q), target, 32.0)
            edge(2, unit(channel, q), target, 1.0)
            for p in orth[q]:
                edge(2, unit(channel, p), target, 1.0)

# ---------------------------------------------------------------------------
# h3:
# 2/3 graded short-range vulnerability.
# 4/5 further connected safety evidence. The weaker second term also recognizes
# distributed liberties, without treating it as exact group-liberty counting.
# ---------------------------------------------------------------------------
copy_channel(3, 0, 0)
copy_channel(3, 1, 1)
for q in range(N):
    for color in (0, 1):
        danger = unit(2 + color, q)
        bias(3, danger, -5.5)
        edge(3, unit(color, q), danger, 8.0)
        edge(3, unit(2 + color, q), danger, -1.0)

        safety = unit(4 + color, q)
        bias(3, safety, -128.25)
        edge(3, unit(color, q), safety, 128.0)
        edge(3, unit(2 + color, q), safety, 0.25)
        edge(3, unit(4 + color, q), safety, 0.5)
        for p in orth[q]:
            edge(3, unit(4 + color, p), safety, 0.5)

# ---------------------------------------------------------------------------
# h4:
# 0/1 occupancy, 2/3 safety-discounted vulnerability,
# 4/5 smooth own/opponent territorial dominance.
# ---------------------------------------------------------------------------
copy_channel(4, 0, 0)
copy_channel(4, 1, 1)
for q in range(N):
    for color in (0, 1):
        target = unit(2 + color, q)
        edge(4, unit(2 + color, q), target, 1.0)
        edge(4, unit(4 + color, q), target, -0.55)
        for p in orth[q]:
            edge(4, unit(4 + color, p), target, -0.55)

sigma2 = 1.45 ** 2
full_mass = sum(
    np.exp(-(dr*dr + dc*dc) / (2.0*sigma2))
    for dr in range(-5, 6)
    for dc in range(-5, 6)
    if dr*dr + dc*dc <= 25
)

for q, (r, c) in enumerate(coords):
    kernel = []
    for p, (rr, cc) in enumerate(coords):
        d2 = (r-rr)**2 + (c-cc)**2
        value = np.exp(-d2 / (2.0*sigma2)) if d2 <= 25 else 0.0
        kernel.append(value)

    boundary_factor = min(1.65, full_mass / sum(kernel))
    own_dom = unit(4, q)
    opp_dom = unit(5, q)
    bias(4, own_dom, -2.1)
    bias(4, opp_dom, -2.1)
    for p, value in enumerate(kernel):
        if value == 0.0:
            continue
        value *= boundary_factor
        edge(4, unit(0, p), own_dom, value)
        edge(4, unit(1, p), own_dom, -value)
        edge(4, unit(1, p), opp_dom, value)
        edge(4, unit(0, p), opp_dom, -value)

for layer in (2, 3, 4):
    for g in GLOBALS:
        edge(layer, g, g, 1.0)

# ---------------------------------------------------------------------------
# h5 candidate features:
# 0 strategic score, offset to keep it nonnegative
# 1 adjacent friendly vulnerability
# 2 adjacent opposing vulnerability
# 3 raw self-atari risk
# 4 eye-filling penalty
# 5 unsupported enemy contact
# ---------------------------------------------------------------------------
SCORE_OFFSET = 64.0

own_shape = {
    1: 0.35, 2: 0.75, 4: 0.75, 5: 0.44,
    8: 0.22, 9: 0.17, 10: 0.08, 13: 0.03,
}
opp_shape = {
    1: -0.65, 2: -0.05, 4: 0.55, 5: 0.34,
    8: 0.20, 9: 0.14, 10: 0.10, 13: 0.04,
}
opening_by_line = [-4.5, -1.2, 1.8, 2.2, 3.3]

for q, (r, c) in enumerate(coords):
    degree = len(orth[q])
    score = unit(0, q)
    save_pressure = unit(1, q)
    capture_pressure = unit(2, q)
    risk = unit(3, q)
    eye = unit(4, q)
    contact = unit(5, q)

    bias(5, score, SCORE_OFFSET + 0.35)
    line = min(r, c, 8-r, 8-c)
    edge(5, OPENING, score, opening_by_line[line])
    edge(5, OWN_COUNT, score, 0.008)
    edge(5, OPP_COUNT, score, 0.008)

    for p, (rr, cc) in enumerate(coords):
        d2 = (r-rr)**2 + (c-cc)**2
        # Broad crowding discourages repeatedly developing one small region.
        crowding = -0.08 * np.exp(-d2 / 18.0)
        edge(5, unit(0, p), score, own_shape.get(d2, 0.0) + crowding)
        edge(5, unit(1, p), score, opp_shape.get(d2, 0.0) + crowding)

    edge(5, unit(4, q), score, -1.05)
    edge(5, unit(5, q), score, -1.25)

    # Empty-neighbor liberties plus estimated support from friendly groups.
    bias(5, risk, 2.0 - degree)
    bias(5, contact, -0.25)

    interior = degree == 4
    bias(5, eye, (1.4 if interior else 1.0) - 2.0*degree)

    for p in orth[q]:
        edge(5, unit(2, p), save_pressure, 1.0)
        edge(5, unit(3, p), capture_pressure, 1.0)

        edge(5, unit(0, p), risk, -1.0)
        edge(5, unit(1, p), risk, 1.0)
        edge(5, unit(2, p), risk, 4.0/3.0)

        edge(5, unit(0, p), eye, 2.0)
        edge(5, unit(1, p), contact, 1.0)
        edge(5, unit(0, p), contact, -0.9)

    for p in diag[q]:
        edge(5, unit(1, p), eye, -1.0 if interior else -1.25)
        edge(5, unit(0, p), contact, -0.45)

for g in GLOBALS:
    edge(5, g, g, 1.0)

# A deliberately modest, approximate score signal used only for passing.
LEAD_POS = 491
LEAD_NEG = 492
for target, sign in ((LEAD_POS, 1.0), (LEAD_NEG, -1.0)):
    bias(5, target, -7.5 * sign)
    edge(5, OWN_COUNT, target, sign)
    edge(5, OPP_COUNT, target, -sign)
    edge(5, WHITE, target, 15.0 * sign)
    for q in range(N):
        edge(5, unit(4, q), target, 0.12 * sign)
        edge(5, unit(5, q), target, -0.12 * sign)

# ---------------------------------------------------------------------------
# h6: combine tactical conditions.
# Rescuing by merely moving an atari group into another atari is suppressed.
# Captures can relieve self-atari and unsupported-contact penalties.
# ---------------------------------------------------------------------------
for q in range(N):
    edge(6, unit(0, q), unit(0, q), 1.0)

    edge(6, unit(1, q), unit(1, q), 1.0)
    edge(6, unit(3, q), unit(1, q), -3.5)
    edge(6, unit(4, q), unit(1, q), -3.0)

    edge(6, unit(2, q), unit(2, q), 1.0)

    edge(6, unit(3, q), unit(3, q), 1.0)
    edge(6, unit(2, q), unit(3, q), -1.15)

    edge(6, unit(4, q), unit(4, q), 1.0)

    edge(6, unit(5, q), unit(5, q), 1.0)
    edge(6, unit(2, q), unit(5, q), -0.9)

for g in GLOBALS:
    edge(6, g, g, 1.0)

# Two-ReLU differences implement bounded passing adjustments.
WIN_LOW, WIN_HIGH, LOSS_LOW, LOSS_HIGH = 491, 492, 493, 494
for target, source, threshold in (
    (WIN_LOW, LEAD_POS, 2.0),
    (WIN_HIGH, LEAD_POS, 12.0),
    (LOSS_LOW, LEAD_NEG, 2.0),
    (LOSS_HIGH, LEAD_NEG, 12.0),
):
    edge(6, source, target, 1.0)
    bias(6, target, -threshold)

# ---------------------------------------------------------------------------
# Linear policy head. Occupancy, suicide and simple ko use the supplied mask.
# ---------------------------------------------------------------------------
gain = 1.30
feature_weights = (1.0, 2.1, 2.8, -3.8, -4.5, -1.3)
for q in range(N):
    bias(7, q, -gain * SCORE_OFFSET)
    for channel, weight in enumerate(feature_weights):
        edge(7, unit(channel, q), q, gain * weight)

bias(7, PASS, -gain)
edge(7, EARLY_PASS, PASS, -0.40 * gain)
edge(7, WIN_LOW, PASS, 0.08 * gain)
edge(7, WIN_HIGH, PASS, -0.08 * gain)
edge(7, LOSS_LOW, PASS, -0.12 * gain)
edge(7, LOSS_HIGH, PASS, 0.12 * gain)

# Export the complete candidate. No network evaluations are performed.
for k in range(1, 8):
    save(f"W{k}", W[k-1])
    save(f"b{k}", B[k-1])

print("Constructed symmetric liberty, safety, shape, territory and passing circuits.")
print("Nonzero coefficients:",
      sum(np.count_nonzero(a) for a in W + B))
