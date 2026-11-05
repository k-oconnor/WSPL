import numpy as np

# Construction only: no positions, simulation, or network evaluations.
W1 = load("W1")
b1 = load("b1")
W2 = load("W2")
W3 = load("W3")

def u(point, channel):
    return 6 * point + channel

coords = [(i // 9, i % 9) for i in range(81)]
neighbors = []
for r, c in coords:
    neighbors.append([
        9*rr + cc
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
        if 0 <= rr < 9 and 0 <= cc < 9
    ])

# Recover the existing supplemental three-liberty destinations.
full_sites = [
    i for i, (r, c) in enumerate(coords)
    if (r + c) % 2 == 0 or r == 4 or c == 4
]
spare2 = [u(i, 1) for i in range(81)] + list(range(494, 512))
assert len(full_sites) == 49

full_three = {}
for index, point in enumerate(full_sites):
    for color in (0, 1):
        full_three[(point, color)] = spare2[2*index + color]

corners = [(0, 0), (0, 8), (8, 0), (8, 8)]
own_seed_slots = [496, 497, 498, 499]
opp_seed_slots = [492, 493, 494, 495]

for (r, c), own_seed, opp_seed in zip(
        corners, own_seed_slots, opp_seed_slots):
    point = 9*r + c
    inward_r = 1 if r == 0 else 7
    inward_c = 1 if c == 0 else 7

    stones = {
        point,
        9*inward_r + c,
        9*r + inward_c,
    }
    boundary = set()
    for j in stones:
        boundary.update(neighbors[j])
    boundary.difference_update(stones)
    assert len(stones) == 3 and len(boundary) == 3

    assert not np.any(W1[:, own_seed])
    assert b1[own_seed] == 0
    assert not np.any(W2[own_seed, :])

    # At a friendly corner L, L counts distinct empty boundary
    # points. Otherwise the missing-stone gate forces zero.
    # seed = relu(L - 1 - 4*missing_friendly_stones).
    b1[own_seed] = -10
    for j in stones:
        W1[3*j, own_seed] = 4
    for j in boundary:
        W1[3*j, own_seed] = -1
        W1[3*j + 1, own_seed] = -1

    # Corner directional channels are consumed only through their
    # sum; neither has an individual directional-three decoder.
    assert np.array_equal(
        W3[u(point, 2), :],
        W3[u(point, 4), :]
    )

    # Inject the distinct-liberty seed into one directional bound.
    #
    # For a corner L with exactly two boundary liberties, the
    # resulting sum of directional certificates is at most two.
    # Thus this establishes two liberties without generating a
    # false three-liberty certificate. With three liberties, the
    # sum exceeds two. The summed result is axis-symmetric.
    W2[own_seed, u(point, 2)] += np.float32(1)

    # Strengthen the supplemental three-liberty certificate.
    # Its original bound at the corner is half the two adjacent
    # stones' direct-liberty counts.
    #
    # If L=2, that bound is at most 1.5, so adding seed/2 cannot
    # exceed the existing threshold of two. If L=3, the bound
    # is two and seed/2 is one, producing a sound certificate.
    W2[own_seed, full_three[(point, 0)]] += np.float32(0.5)
    W2[opp_seed, full_three[(point, 1)]] += np.float32(0.5)

save("W1", W1)
save("b1", b1)
save("W2", W2)

print("Added four friendly corner distinct-liberty certificates.")
print("Strengthened corner three-liberty certificates for both colors.")
print("Preserved all positional, tactical-scoring, and passing weights.")
