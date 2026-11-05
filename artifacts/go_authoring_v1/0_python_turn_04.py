import numpy as np

# Coefficient construction only: no positions or network evaluations.
W = {k: load("W" + str(k)) for k in range(1, 5)}
B = {k: load("b" + str(k)) for k in range(1, 5)}

def u(point, channel):
    return 6 * point + channel

neighbors = []
for r in range(9):
    for c in range(9):
        neighbors.append([
            9*rr + cc
            for rr, cc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1))
            if 0 <= rr < 9 and 0 <= cc < 9
        ])

# h1 slots are unused. The intermediate slots contain obsolete
# color/material copies, or are unused. Area estimation already
# consumes the original h1 material counts directly.
input_slots = [492, 493, 494, 495]
aux_slots = [489, 490, 491, 511]

for source in input_slots:
    assert not np.any(W[1][:, source])
    assert B[1][source] == 0
    assert not np.any(W[2][source, :])

# Check that these intermediate carriers have no existing consumers
# except their own unused carry chains.
for layer in (3, 4):
    for slot in aux_slots:
        row = W[layer][slot, :].copy()
        row[slot] = 0
        assert not np.any(row)

for layer in (5, 6):
    matrix = load("W" + str(layer))
    for slot in aux_slots:
        row = matrix[slot, :].copy()
        row[slot] = 0
        assert not np.any(row)

output = load("W7")
assert not np.any(output[aux_slots, :])

corners = [(0,0), (0,8), (8,0), (8,8)]

for (r, c), seed, aux in zip(corners, input_slots, aux_slots):
    inward_r = 1 if r == 0 else 7
    inward_c = 1 if c == 0 else 7
    stones = {
        9*r + c,
        9*inward_r + c,
        9*r + inward_c,
    }

    boundary = set()
    for point in stones:
        boundary.update(neighbors[point])
    boundary.difference_update(stones)
    assert len(stones) == 3
    assert len(boundary) == 3

    # When all three connected stones are opposing stones,
    # L is the number of distinct empty boundary points.
    # Each such point is a genuine liberty of their group.
    #
    # seed = relu(L - 1 - 4*missing_opposing_stones).
    # Thus a positive seed certifies at least two liberties,
    # without double-counting the shared diagonal liberty.
    W[1][:, seed] = 0
    B[1][seed] = len(boundary) - 1 - 4*len(stones)
    for point in stones:
        W[1][3*point + 1, seed] = 4
    for point in boundary:
        W[1][3*point, seed] = -1
        W[1][3*point + 1, seed] = -1

    # Preserve the integer-valued two-liberty seed through h2.
    W[2][:, aux] = 0
    B[2][aux] = 0
    W[2][seed, aux] = 1

    # Its excess above one certifies at least three liberties.
    W[3][:, aux] = 0
    B[3][aux] = -1
    W[3][aux, aux] = 1

    # Existing destination gates require an opposing stone.
    # Any opposing stone here is connected to the certified triple.
    attachment = stones | boundary
    for point in attachment:
        W[3][aux, u(point, 3)] += np.float32(1)
        W[4][aux, u(point, 5)] += np.float32(1)

for layer in range(1, 5):
    save("W" + str(layer), W[layer])
    save("b" + str(layer), B[layer])

print("Added four opposing corner L-shape liberty certificates.")
print("Preserved existing certificates, area estimation, and move scoring.")
