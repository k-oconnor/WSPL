import numpy as np

W1, b1 = load("W1"), load("b1")
W2 = load("W2")
W5, b5 = load("W5"), load("b5")
W6 = load("W6")

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

def boundary(region):
    inside = set(region)
    outside = set()
    for s in inside:
        outside.update(neighbors[s])
    return sorted(outside - inside)

# Eight connected patterns, invariant as a collection under the
# board's rotations and reflections.
small_regions = []

# Solid blocks spanning the second and third lines near each corner.
for r0 in (1, 6):
    for c0 in (1, 6):
        small_regions.append([
            9*r + c
            for r in range(r0, r0 + 2)
            for c in range(c0, c0 + 2)
        ])

# Three-stone strings near the middle of each side.
for fixed in (1, 7):
    small_regions.append([9*fixed + c for c in range(3, 6)])
    small_regions.append([9*r + fixed for r in range(3, 6)])

assert len(small_regions) == 8
auxiliary = list(range(496, 512))

# These H1 producers and their H2 outgoing rows were unused.
assert np.count_nonzero(W1[:, auxiliary]) == 0
assert np.count_nonzero(b1[auxiliary]) == 0
assert np.count_nonzero(W2[auxiliary, :]) == 0

# For a connected required-stone region S with external boundary B:
#
# U_k = relu(|B| - k + 1
#            - sum_{t in B} occupied(t)
#            - |B| * missing_required_stones).
#
# If any required stone is missing, U_k is zero for k >= 2.
# Otherwise U_k = relu(distinct_boundary_liberties - k + 1).
# Thus U_k >= 1 certifies at least k liberties for the whole group.
#
# H1: two-liberty certificates for both colors.
for region_index, region in enumerate(small_regions):
    outside = boundary(region)
    footprint = sorted(set(region) | set(outside))
    gate = len(outside)

    for color in (0, 1):
        unit = 496 + 2*region_index + color

        b1[unit] = len(outside) - 1 - gate*len(region)
        for s in region:
            W1[3*s + color, unit] = gate
        for t in outside:
            W1[3*t, unit] = -1
            W1[3*t + 1, unit] = -1

        # Existing H2 two-liberty deficits are at most three.
        # Subtracting three times a certificate sets them to zero
        # throughout the connected region and its same-color neighbors.
        for s in footprint:
            W2[unit, ch(2 + color, s)] = -3

# H5 has another sixteen unused producers. Use them for prospective
# move safety, covering the eight new patterns plus the eight existing
# five-stone third-line segments.
support_regions = [list(region) for region in small_regions]
for fixed in (2, 6):
    for start in (0, 4):
        support_regions.append([
            9*fixed + c for c in range(start, start + 5)
        ])
        support_regions.append([
            9*r + fixed for r in range(start, start + 5)
        ])

assert len(support_regions) == 16
assert np.count_nonzero(W5[:, auxiliary]) == 0
assert np.count_nonzero(b5[auxiliary]) == 0
assert np.count_nonzero(W6[auxiliary, :]) == 0

for unit, region in zip(auxiliary, support_regions):
    outside = boundary(region)
    gate = len(outside)

    # H4 channels 0 and 1 still carry exact stone occupancy.
    # This certificate concerns friendly stones only.
    b5[unit] = len(outside) - 2 - gate*len(region)
    for s in region:
        W5[ch(0, s), unit] = gate
    for t in outside:
        W5[ch(0, t), unit] = -1
        W5[ch(1, t), unit] = -1

    # A legal placement on this boundary joins a friendly group
    # with at least three liberties and consumes at most one.
    # The existing prospective-liberty deficit is bounded above by
    # two, so this subtraction removes only a provably false penalty.
    for t in outside:
        W6[unit, ch(3, t)] = -2

for name, array in (
    ("W1", W1), ("b1", b1), ("W2", W2),
    ("W5", W5), ("b5", b5), ("W6", W6),
):
    assert np.isfinite(array).all()
    save(name, array)

print("Added 16 early boundary-liberty certificates.")
print("Added 16 certified friendly-support circuits.")
print("No forward passes, fitting, search, or gameplay queries.")
