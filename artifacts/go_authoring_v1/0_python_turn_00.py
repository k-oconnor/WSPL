import numpy as np

# Construction only: no positions, network evaluations, search, or fitting.
widths = [243, 512, 512, 512, 512, 512, 512, 82]
W = {}
B = {}
for layer in range(1, 8):
    W[layer] = np.zeros((widths[layer-1], widths[layer]), dtype=np.float32)
    B[layer] = np.zeros(widths[layer], dtype=np.float32)

def add(layer, source, destination, coefficient):
    W[layer][source, destination] += np.float32(coefficient)

def unit(point, channel):
    return 6 * point + channel

coords = [(i // 9, i % 9) for i in range(81)]
neighbors = []
for r, c in coords:
    neighbors.append([
        9 * rr + cc
        for rr, cc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
        if 0 <= rr < 9 and 0 <= cc < 9
    ])

# Symmetric, bounded-radius geometric kernels.
territory_kernel = {
    (0,1): 1.00,
    (1,1): 0.65,
    (0,2): 0.50,
    (1,2): 0.25,
    (0,3): 0.15,
    (2,2): 0.15,
}
nearby = []
for i, (r, c) in enumerate(coords):
    entries = []
    for j, (rr, cc) in enumerate(coords):
        key = tuple(sorted((abs(rr-r), abs(cc-c))))
        if key in territory_kernel:
            entries.append((j, key))
    nearby.append(entries)

EARLY = 486
MIDDLE = 487
PASS_EARLY = 488
WHITE = 489
OWN_COUNT = 490
OPP_COUNT = 491
ADV_POS = 492
ADV_NEG = 493
ADV_POS_EXCESS = 494
ADV_NEG_EXCESS = 495
THREATS = 496
THREATS_EXCESS = 497

# h1 point channels:
# 0 own, 1 opponent, 2 own direct-liberty count,
# 3 opponent direct-liberty count,
# 4/5 empty-point territorial influence of each color.
for i in range(81):
    add(1, 3*i,   unit(i,0), 1)
    add(1, 3*i+1, unit(i,1), 1)

    degree = len(neighbors[i])
    for color, channel in [(0,2), (1,3)]:
        destination = unit(i,channel)
        B[1][destination] = degree - 4
        add(1, 3*i+color, destination, 4)
        for j in neighbors[i]:
            add(1, 3*j,   destination, -1)
            add(1, 3*j+1, destination, -1)

    normalizer = sum(territory_kernel[key] for j, key in nearby[i])
    for color, channel in [(0,4), (1,5)]:
        destination = unit(i,channel)
        add(1, 3*i,   destination, -2)
        add(1, 3*i+1, destination, -2)
        for j, key in nearby[i]:
            coefficient = territory_kernel[key] / normalizer
            add(1, 3*j+color,   destination, coefficient)
            add(1, 3*j+1-color, destination, -coefficient)

# Board-density schedules and color-relative material.
for i in range(81):
    for color in (0,1):
        source = 3*i+color
        add(1, source, EARLY, -1/28)
        add(1, source, MIDDLE, -1/62)
        add(1, source, PASS_EARLY, -1/20)
    add(1, 3*i, OWN_COUNT, 1)
    add(1, 3*i+1, OPP_COUNT, 1)
B[1][EARLY] = 1
B[1][MIDDLE] = 1
B[1][PASS_EARLY] = 1
add(1, 2, WHITE, 1)

carried_globals = [EARLY, MIDDLE, PASS_EARLY, WHITE, OWN_COUNT, OPP_COUNT]
for layer in range(2, 7):
    for g in carried_globals:
        add(layer, g, g, 1)

# h2 point channels:
# 0/1 occupancy; 2/3 proofs of at least two group liberties;
# 4/5 proofs of at least three group liberties.
#
# At an occupied point, its direct liberties are disjoint from those
# of its adjacent stones. A liberty can be counted by at most two
# adjacent stones, so half their summed counts is a sound lower bound.
for i in range(81):
    for color in (0,1):
        add(2, unit(i,color), unit(i,color), 1)
        direct_channel = 2 + color
        for threshold, out_channel in [(1,2+color), (2,4+color)]:
            destination = unit(i,out_channel)
            B[2][destination] = -16 - threshold
            add(2, unit(i,color), destination, 16)
            add(2, unit(i,direct_channel), destination, 1)
            for j in neighbors[i]:
                add(2, unit(j,direct_channel), destination, 0.5)

# Approximate area advantage, including the correct relative komi.
for destination, sign in [(ADV_POS,1), (ADV_NEG,-1)]:
    B[2][destination] = -7.5 * sign
    add(2, OWN_COUNT, destination, sign)
    add(2, OPP_COUNT, destination, -sign)
    add(2, WHITE, destination, 15 * sign)
    for i in range(81):
        add(2, unit(i,4), destination, sign)
        add(2, unit(i,5), destination, -sign)

# Propagate liberty proofs only through stones of the same color.
# These gate constants exceed the corresponding possible incoming sums.
for layer, gate in [(3,32), (4,128)]:
    for i in range(81):
        for color in (0,1):
            add(layer, unit(i,color), unit(i,color), 1)
        for channel, color in [(2,0), (3,1), (4,0), (5,1)]:
            destination = unit(i,channel)
            B[layer][destination] = -gate
            add(layer, unit(i,color), destination, gate)
            for j in [i] + neighbors[i]:
                add(layer, unit(j,channel), destination, 1)

# Clip the area estimate to keep it a modest passing adjustment.
for source, excess in [(ADV_POS,ADV_POS_EXCESS),
                       (ADV_NEG,ADV_NEG_EXCESS)]:
    add(3, source, source, 1)
    add(3, source, excess, 1)
    B[3][excess] = -20
    add(4, source, source, 1)
    add(4, excess, source, -1)
    add(5, source, source, 1)
    add(6, source, source, 1)

# h5: occupancy and binary absence-of-proof indicators.
# Positive proof values are multiples of 1/2, so this yields exact
# binary flags on valid binary board inputs.
for i in range(81):
    for color in (0,1):
        add(5, unit(i,color), unit(i,color), 1)
    for channel, color in [(2,0), (3,1), (4,0), (5,1)]:
        destination = unit(i,channel)
        add(5, unit(i,color), destination, 1)
        for j in [i] + neighbors[i]:
            add(5, unit(j,channel), destination, -2)

base_own = {
    (0,1): -0.85,
    (1,1):  0.22,
    (0,2):  0.60,
    (1,2):  0.24,
    (0,3):  0.15,
    (2,2):  0.08,
}
base_opp = {
    (0,1): -0.35,
    (1,1):  0.18,
    (0,2):  0.32,
    (1,2):  0.20,
    (0,3):  0.12,
    (2,2):  0.08,
}
waste_own = {
    (0,1): 1.00,
    (1,1): 0.50,
    (0,2): 0.28,
    (1,2): 0.16,
    (0,3): 0.08,
    (2,2): 0.08,
}
waste_opp = {
    (0,1): -1.30,
    (1,1): -0.65,
    (0,2): -0.45,
    (1,2): -0.25,
    (0,3): -0.12,
    (2,2): -0.10,
}
danger_own = {
    (0,1): -1.05,
    (1,1): -0.60,
    (0,2): -0.33,
    (1,2): -0.20,
    (0,3): -0.09,
    (2,2): -0.08,
}
danger_opp = {
    (0,1): 0.95,
    (1,1): 0.45,
    (0,2): 0.23,
    (1,2): 0.13,
    (0,3): 0.06,
    (2,2): 0.06,
}

opening_prior = [-4.0, -2.4, 0.2, 0.4, 0.7]
middle_prior  = [-0.8, -0.2, 0.3, 0.3, 0.25]
static_prior  = [-0.45, -0.10, 0.0, 0.0, 0.0]

# h6 point channels:
# 0 shifted affine move score
# 1 excess tactical urgency (caps repeated counting)
# 2 self-atari penalty
# 3 friendly-eye penalty
# 4 overconcentration / friendly-territory penalty
# 5 unsupported hostile-contact / invasion penalty
for i, (r,c) in enumerate(coords):
    degree = len(neighbors[i])
    edge_distance = min(r,c,8-r,8-c)
    base, cap, self_atari, eye, waste, danger = [
        unit(i,k) for k in range(6)
    ]

    B[6][base] = 34 + static_prior[edge_distance]
    add(6, EARLY, base, opening_prior[edge_distance])
    add(6, MIDDLE, base, middle_prior[edge_distance])

    B[6][cap] = -10
    B[6][self_atari] = 2 - degree
    B[6][eye] = 1 - degree
    B[6][waste] = -1.55 + 0.30*(4-degree)
    add(6, EARLY, waste, -0.25)
    B[6][danger] = -0.60
    add(6, EARLY, danger, 0.15)

    for j, key in nearby[i]:
        add(6, unit(j,0), base, base_own[key])
        add(6, unit(j,1), base, base_opp[key])
        add(6, unit(j,0), waste, waste_own[key])
        add(6, unit(j,1), waste, waste_opp[key])
        add(6, unit(j,0), danger, danger_own[key])
        add(6, unit(j,1), danger, danger_opp[key])

    for j in neighbors[i]:
        # Urgency: rescuing atari, capturing, and two-liberty pressure.
        for channel, coefficient in [(2,5.5), (3,6.5), (4,1.0), (5,1.8)]:
            add(6, unit(j,channel), base, coefficient)
            add(6, unit(j,channel), cap, coefficient)

        # Liberty budget:
        # empty neighbors + 2*strong friendly contacts
        # + .75*weak non-atari friendly contacts + capture allowance.
        add(6, unit(j,0), self_atari, -1)
        add(6, unit(j,1), self_atari,  1)
        add(6, unit(j,4), self_atari,  1.25)
        add(6, unit(j,2), self_atari,  0.75)
        add(6, unit(j,3), self_atari, -3)

        # Preserve eyes except when joining an endangered friendly group.
        add(6, unit(j,0), eye,  1)
        add(6, unit(j,2), eye, -1)

        # Rescue and capture exceptions to positional caution.
        add(6, unit(j,2), waste, -1.6)
        add(6, unit(j,5), danger, -1.1)
        add(6, unit(j,3), danger, -2.0)
        add(6, unit(j,2), danger, -0.6)

    add(7, base,       i,   1.0)
    add(7, cap,        i,  -1.0)
    add(7, self_atari, i, -14.0)
    add(7, eye,        i, -12.0)
    add(7, waste,      i,  -3.2)
    add(7, danger,     i,  -2.5)
    B[7][i] = -32

# Passing becomes plausible only after useful moves have dried up.
for i in range(81):
    for channel in (2,3):
        add(6, unit(i,channel), THREATS, 1)
        add(6, unit(i,channel), THREATS_EXCESS, 1)
B[6][THREATS_EXCESS] = -6

B[7][81] = -1.7
add(7, PASS_EARLY, 81, -7.0)
add(7, MIDDLE, 81, -1.2)
add(7, ADV_POS, 81, 0.05)
add(7, ADV_NEG, 81, -0.05)
add(7, THREATS, 81, -0.12)
add(7, THREATS_EXCESS, 81, 0.12)

for layer in range(1,8):
    save("W" + str(layer), W[layer])
    save("b" + str(layer), B[layer])

print("Installed explicit liberty-proof, tactical, shape, and passing circuits.")
print("Coefficient count:",
      sum(W[k].size + B[k].size for k in range(1,8)))
