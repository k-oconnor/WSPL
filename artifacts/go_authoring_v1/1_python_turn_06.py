import numpy as np

W = {k: load(f"W{k}") for k in (2, 3, 4)}
B = {k: load(f"b{k}") for k in (2, 3, 4)}

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

# Match the existing H2-H4 auxiliary chain circuits.
segments = []
for fixed in (2, 6):
    for start in (0, 4):
        segments.append([
            9 * fixed + c for c in range(start, start + 5)
        ])
        segments.append([
            9 * r + fixed for r in range(start, start + 5)
        ])

# Match the existing H1 auxiliary two-liberty certificates.
small_regions = []
for r0 in (1, 6):
    for c0 in (1, 6):
        small_regions.append([
            9*r + c
            for r in range(r0, r0 + 2)
            for c in range(c0, c0 + 2)
        ])

for fixed in (1, 7):
    small_regions.append([9*fixed + c for c in range(3, 6)])
    small_regions.append([9*r + fixed for r in range(3, 6)])

small_boundaries = [boundary(region) for region in small_regions]

assert len(segments) == len(small_regions) == 8

# Replace only the auxiliary producers. Their existing connections
# into the ordinary safety-complement channels remain intact.
for layer in (2, 3, 4):
    W[layer][:, 496:512] = 0
    B[layer][496:512] = 0

for segment_index, segment in enumerate(segments):
    outside = boundary(segment)
    footprint = sorted(set(segment) | set(outside))
    footprint_set = set(footprint)
    n = len(segment)

    # Pure geometry check: an external point touches exactly one
    # stone of this straight segment. Thus direct liberties summed
    # over a fully occupied segment are distinct.
    assert all(
        sum(t in neighbors[s] for s in segment) == 1
        for t in outside
    )

    linked_patterns = [
        i for i, region in enumerate(small_regions)
        if footprint_set.intersection(region)
    ]

    for color in (0, 1):
        unit = 496 + 2*segment_index + color

        # H2-H3: binary two-liberty certificate.
        #
        # With the whole segment occupied by this color, any of:
        #   - two direct liberties of the segment;
        #   - a bordering friendly stone with >=2 direct liberties;
        #   - an adjoining active H1 two-liberty pattern;
        # proves that the segment's group has >=2 liberties.
        #
        # F = sum_segment L
        #     + 2*sum_boundary relu(L-1)
        #     + 2*sum_linked_pattern_certificates.
        #
        # Every quantity is a nonnegative integer. Hence F >= 2
        # implies at least one of these valid proofs.
        pattern_bound = sum(
            len(small_boundaries[i]) - 1
            for i in linked_patterns
        )
        missing_gate = (
            4*n + 6*len(outside) + 2*pattern_bound
        )

        # D = relu(2 - F + missing_gate*missing_stones).
        # The gate bounds F even on an incomplete segment.
        B[2][unit] = 2 + missing_gate*n

        for s in segment:
            W[2][ch(color, s), unit] = -missing_gate
            W[2][ch(2 + color, s), unit] = -1

        for t in outside:
            W[2][ch(4 + color, t), unit] = -2

        for pattern_index in linked_patterns:
            pattern_unit = 496 + 2*pattern_index + color
            W[2][pattern_unit, unit] = -2

        # U = relu(1-D), exactly zero or one.
        W[3][unit, unit] = -1
        B[3][unit] = 1

        # Existing H4 outgoing uses of U subtract it from two-liberty
        # deficits throughout the segment and its immediate boundary.

        # H4: independent three-liberty certificate.
        #
        # H3 still carries exact occupancy and binary local
        # three-liberty certificates. On a complete segment:
        #
        # R = relu(boundary_empty_count - 2
        #          + 3*sum_footprint local_three_liberty_certificate).
        #
        # R >= 1 proves >=3 liberties, either directly or through
        # any adjoining same-color stone already certified safe.
        #
        # The completeness gate dominates the entire positive
        # expression whenever any required segment stone is missing.
        complete_gate = len(outside) + 3*len(footprint)
        B[4][unit] = (
            len(outside) - 2 - complete_gate*n
        )

        for s in segment:
            W[4][ch(color, s), unit] = complete_gate

        for t in outside:
            W[4][ch(0, t), unit] = -1
            W[4][ch(1, t), unit] = -1

        for t in footprint:
            W[4][ch(4 + color, t), unit] = 3

        # Existing H5 connections use positive R to remove both
        # two- and three-liberty deficits on this connected footprint.
        # All previous direct-boundary and internal certificates
        # remain sufficient under the replacement formulas.

for k in (2, 3, 4):
    assert np.isfinite(W[k]).all()
    assert np.isfinite(B[k]).all()
    save(f"W{k}", W[k])
    save(f"b{k}", B[k])

print("Extended connected-chain liberty certificates.")
print("Scoring weights unchanged.")
print("Coefficient construction only; no forward passes or evaluation.")
