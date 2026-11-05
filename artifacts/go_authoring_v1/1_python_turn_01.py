import numpy as np

W = {k: load(f"W{k}") for k in range(1, 8)}
B = {k: load(f"b{k}") for k in range(1, 8)}

def ch(channel, point):
    return 81 * channel + point

def add(layer, source, destination, value):
    W[layer][source, destination] += np.float32(value)

def clear_column(layer, destination):
    W[layer][:, destination] = 0
    B[layer][destination] = 0

neighbors = []
for r in range(9):
    for c in range(9):
        neighbors.append([
            9*rr + cc
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if 0 <= rr < 9 and 0 <= cc < 9
        ])

scale = np.float32(1.20)

# These are coefficient transformations, not network evaluations.
shape_coefficients = W[7][:162, :81].copy() / scale
own_enclosure_coefficients = W[1][:, 324:405].copy()
opp_enclosure_coefficients = W[1][:, 405:486].copy()
own_enclosure_bias = B[1][324:405].copy()
opp_enclosure_bias = B[1][405:486].copy()

# Replace the intermediate enclosure channel with opposing three-liberty
# certification. A stone's local proof statistic is
# P = 2*direct_liberties + sum(adjacent friendly direct_liberties).
# Each distinct liberty contributes at most twice, so P >= 5 proves >=3.
for s in range(81):
    dest = ch(5, s)

    clear_column(2, dest)
    add(2, ch(1, s), dest, 5)
    B[2][dest] = -2 * len(neighbors[s])
    for t in neighbors[s]:
        add(2, ch(0, t), dest, 2)
        add(2, ch(1, t), dest, 2)
        add(2, ch(3, t), dest, -1)

    # Convert the integer deficit into a binary safety certificate.
    clear_column(3, dest)
    add(3, ch(1, s), dest, 1)
    add(3, ch(5, s), dest, -1)

    # Complement of certification propagated across one friendly edge.
    clear_column(4, dest)
    add(4, ch(1, s), dest, 1)
    add(4, ch(5, s), dest, -1)
    for t in neighbors[s]:
        add(4, ch(5, t), dest, -1)

    # Propagate certification across a second friendly edge.
    clear_column(5, dest)
    add(5, ch(5, s), dest, 1)
    for t in neighbors[s]:
        add(5, ch(1, t), dest, -1)
        add(5, ch(5, t), dest, 1)

# H6 no longer needs separate own/opponent occupancy copies.
# Its revised channels are:
# 0: shifted affine shape score
# 1: own enclosure
# 2: tactical stone urgency
# 3: existing prospective-liberty deficit
# 4: existing eye penalty
# 5: opposing enclosure
#
# The affine shape score receives an analytically sufficient offset.
# Consequently its ReLU is an identity on all valid board encodings.
W[7][0:162, :81] = 0
W[7][405:486, :81] = 0

own_inputs = 3 * np.arange(81)
opp_inputs = own_inputs + 1

for s in range(81):
    shape_dest = ch(0, s)
    clear_column(6, shape_dest)

    own_shape = shape_coefficients[:81, s]
    opp_shape = shape_coefficients[81:162, s]
    offset = np.float32(
        -np.minimum(np.minimum(own_shape, opp_shape), 0).sum(dtype=np.float64)
        + 0.001
    )

    W[6][:162, shape_dest] = shape_coefficients[:, s]
    B[6][shape_dest] = offset
    W[7][shape_dest, s] = scale
    B[7][s] -= np.float32(scale * offset)

    # Recreate both original enclosure functions directly from H5 stones.
    own_dest = ch(1, s)
    clear_column(6, own_dest)
    W[6][0:81, own_dest] = own_enclosure_coefficients[own_inputs, s]
    W[6][81:162, own_dest] = own_enclosure_coefficients[opp_inputs, s]
    B[6][own_dest] = own_enclosure_bias[s]
    W[7][own_dest, s] = np.float32(-2.10 * scale)

    opp_dest = ch(5, s)
    clear_column(6, opp_dest)
    W[6][0:81, opp_dest] = opp_enclosure_coefficients[own_inputs, s]
    W[6][81:162, opp_dest] = opp_enclosure_coefficients[opp_inputs, s]
    B[6][opp_dest] = opp_enclosure_bias[s]
    W[7][opp_dest, s] = np.float32(-2.85 * scale)

    # Opposing two-liberty uncertainty adds modest attacking pressure.
    # True atari groups necessarily lack both safety certificates.
    # Preserve the previous base urgency for captures and friendly rescues.
    tactical_dest = ch(2, s)
    clear_column(6, tactical_dest)

    size_bonus = 0.35
    gate = 4 * size_bonus
    attack_bonus = 2.60

    add(6, ch(2, s), tactical_dest, 7.50 + gate)
    add(6, ch(3, s), tactical_dest, 12.50 - attack_bonus)
    add(6, ch(5, s), tactical_dest, attack_bonus + gate)
    B[6][tactical_dest] = -gate

    for t in neighbors[s]:
        add(6, ch(2, t), tactical_dest, size_bonus)
        add(6, ch(3, t), tactical_dest, size_bonus)

for k in range(2, 8):
    assert np.isfinite(W[k]).all()
    assert np.isfinite(B[k]).all()
    save(f"W{k}", W[k])
    save(f"b{k}", B[k])

print("Added conservative low-liberty attacking pressure.")
print("Preserved enclosure functions by explicit coefficient factoring.")
print("No forward passes, fitting, search, or gameplay queries.")
