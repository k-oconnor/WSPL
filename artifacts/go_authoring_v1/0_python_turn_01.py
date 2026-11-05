import numpy as np

W = {k: load("W" + str(k)) for k in range(2, 6)}
B = {k: load("b" + str(k)) for k in range(2, 6)}

def u(point, channel):
    return 6 * point + channel

def add(layer, source, destination, coefficient):
    W[layer][source, destination] += np.float32(coefficient)

coords = [(i // 9, i % 9) for i in range(81)]
neighbors = []
for r, c in coords:
    neighbors.append([
        9 * rr + cc
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
        if 0 <= rr < 9 and 0 <= cc < 9
    ])

# Intermediate occupancy encoding:
# opponent = 0, empty = 1, own = 2.
# Own-color gates use G*(occupancy-2).
# Opponent-color gates use -G*occupancy.
for i in range(81):
    for channel in (0, 1):
        destination = u(i, channel)
        W[2][:, destination] = 0
        B[2][destination] = 0
    destination = u(i, 0)
    B[2][destination] = 1
    add(2, u(i, 0), destination, 1)
    add(2, u(i, 1), destination, -1)

# Forty sites, with axes chosen equivariantly under board symmetries.
# At an occupied site, its direct liberties and those of its two
# opposite neighbors are mutually disjoint. Thus their sum is a
# certified lower bound on the group's distinct liberties.
sites = [
    i for i, (r, c) in enumerate(coords)
    if (r + c) % 2 == 1
]
assert len(sites) == 40

extra = {}
slot = 0
for i in sites:
    r, c = coords[i]
    if r % 2 == 0:
        axis = [(r, c-1), (r, c+1)]
    else:
        axis = [(r-1, c), (r+1, c)]
    opposite_neighbors = [
        9 * rr + cc for rr, cc in axis
        if 0 <= rr < 9 and 0 <= cc < 9
    ]

    for color in (0, 1):
        destination = u(slot, 1)
        slot += 1
        extra[(i, color)] = destination
        B[2][destination] = -17
        add(2, u(i, color), destination, 16)
        add(2, u(i, 2 + color), destination, 1)
        for j in opposite_neighbors:
            add(2, u(j, 2 + color), destination, 1)

assert slot == 80

# Rebuild only pointwise intermediate circuits.
# All global schedules and area-estimate circuits remain unchanged.
for layer in (3, 4, 5):
    W[layer][:, :486] = 0
    B[layer][:486] = 0

proof_channels = [(2, 0), (3, 1), (4, 0), (5, 1)]

def gate(layer, point, color, destination, magnitude):
    if color == 0:
        B[layer][destination] = -2 * magnitude
        add(layer, u(point, 0), destination, magnitude)
    else:
        B[layer][destination] = 0
        add(layer, u(point, 0), destination, -magnitude)

# h3: propagate original certificates plus the new two-liberty seeds.
# Conservative gate magnitudes exceed all possible incoming sums.
for i in range(81):
    add(3, u(i, 0), u(i, 0), 1)
    for channel, color in proof_channels:
        destination = u(i, channel)
        gate(3, i, color, destination, 128)
        for j in [i] + neighbors[i]:
            add(3, u(j, channel), destination, 1)
            if channel in (2, 3) and (j, color) in extra:
                add(3, extra[(j, color)], destination, 1)

# A directional two-liberty seed is relu(L-1).
# relu(seed-1) therefore certifies at least three liberties.
for destination in extra.values():
    add(3, destination, destination, 1)
    B[3][destination] = -1

# h4: another same-color propagation step, now also merging the
# additional three-liberty certificates.
for i in range(81):
    add(4, u(i, 0), u(i, 0), 1)
    for channel, color in proof_channels:
        destination = u(i, channel)
        gate(4, i, color, destination, 1024)
        for j in [i] + neighbors[i]:
            add(4, u(j, channel), destination, 1)
            if channel in (4, 5) and (j, color) in extra:
                add(4, extra[(j, color)], destination, 1)

# h5: restore the original interface consumed by the scoring layer:
# own, opponent, absent-own-2, absent-opponent-2,
# absent-own-3, absent-opponent-3.
for i in range(81):
    own = u(i, 0)
    opp = u(i, 1)

    add(5, u(i, 0), own, 1)
    B[5][own] = -1
    add(5, u(i, 0), opp, -1)
    B[5][opp] = 1

    for channel, color in proof_channels:
        destination = u(i, channel)
        if color == 0:
            add(5, u(i, 0), destination, 1)
            B[5][destination] = -1
        else:
            add(5, u(i, 0), destination, -1)
            B[5][destination] = 1

        # Every positive certificate is at least 1/2, making this
        # an exact binary absence-of-certificate indicator.
        for j in [i] + neighbors[i]:
            add(5, u(j, channel), destination, -2)

for layer in range(2, 6):
    save("W" + str(layer), W[layer])
    save("b" + str(layer), B[layer])

print("Added 80 directional two-liberty and 80 three-liberty seed circuits.")
print("Preserved existing tactical, positional, and passing scoring.")
