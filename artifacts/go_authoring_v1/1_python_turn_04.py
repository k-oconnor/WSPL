import numpy as np

W = {k: load(f"W{k}") for k in range(2, 6)}
B = {k: load(f"b{k}") for k in range(2, 6)}

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

# Reflection- and rotation-symmetric straight segments on the third line.
segments = []
for fixed in (2, 6):
    for start in (0, 4):
        segments.append([9 * fixed + c for c in range(start, start + 5)])
        segments.append([9 * r + fixed for r in range(start, start + 5)])

auxiliary = list(range(496, 512))
assert 2 * len(segments) == len(auxiliary)

# These units were unused in the accepted parent.
for layer in (2, 3, 4):
    W[layer][:, auxiliary] = 0
    B[layer][auxiliary] = 0
for layer in (3, 4, 5):
    W[layer][auxiliary, :] = 0

for segment_index, segment in enumerate(segments):
    n = len(segment)

    footprint = set(segment)
    for s in segment:
        footprint.update(neighbors[s])

    for color in (0, 1):
        unit = auxiliary[2 * segment_index + color]

        # When every point of a straight segment is the same color,
        # no empty point can touch two segment stones. Therefore the
        # sum of their direct liberties counts distinct liberties.
        #
        # H2: D = relu(3 - sum(L) + M * missing_stones).
        # M makes incomplete segments ineligible for certification.
        missing_gate = 4 * n
        B[2][unit] = 3 + missing_gate * n

        for s in segment:
            W[2][ch(color, s), unit] = -missing_gate
            W[2][ch(2 + color, s), unit] = -1

        # H3:
        #   0: incomplete segment, or fewer than two direct liberties
        #   1: complete segment with exactly two direct liberties
        #   2: complete segment with at least three direct liberties
        W[3][unit, unit] = -1
        B[3][unit] = 2

        # Any positive value proves two liberties for the segment's
        # group, including same-color stones immediately adjoining it.
        # Existing H4 complements remain binary after subtraction.
        for s in footprint:
            W[4][unit, ch(2 + color, s)] -= np.float32(1)

        # H4: certify three liberties either from the distinct direct
        # liberty count or an existing H3 certificate anywhere along
        # the fully occupied segment.
        #
        # R = relu(U + 2*sum(local_cert3)
        #          - 1 - G*missing_stones).
        #
        # Its positive magnitude is used only as a safety gate.
        complete_gate = 2 * n
        B[4][unit] = -1 - complete_gate * n
        W[4][unit, unit] = 1

        for s in segment:
            W[4][ch(color, s), unit] = complete_gate
            W[4][ch(4 + color, s), unit] = 2

        # A three-liberty certificate also proves two liberties.
        # Restriction to the appropriate color is already enforced
        # by the existing H5 complement circuits.
        for s in footprint:
            W[5][unit, ch(2 + color, s)] -= np.float32(1)
            W[5][unit, ch(4 + color, s)] -= np.float32(1)

for k in range(2, 6):
    assert np.isfinite(W[k]).all()
    assert np.isfinite(B[k]).all()
    save(f"W{k}", W[k])
    save(f"b{k}", B[k])

print("Added 16 straight-chain liberty circuits using spare units.")
print("Existing scoring coefficients unchanged.")
print("No forward passes, fitting, search, or gameplay queries.")
