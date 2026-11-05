import numpy as np

# Coefficient construction only: no positions or network evaluations.
W = {k: load("W" + str(k)) for k in range(2, 7)}
B = {k: load("b" + str(k)) for k in range(2, 7)}

def u(point, channel):
    return 6 * point + channel

def add(layer, source, destination, coefficient):
    W[layer][source, destination] += np.float32(coefficient)

coords = [(i // 9, i % 9) for i in range(81)]
neighbors = []
axes = []
for r, c in coords:
    horizontal = [
        9*r + cc for cc in (c-1, c+1) if 0 <= cc < 9
    ]
    vertical = [
        9*rr + c for rr in (r-1, r+1) if 0 <= rr < 9
    ]
    axes.append((horizontal, vertical))
    neighbors.append(horizontal + vertical)

# Preserve the existing global schedules and area-estimation circuit.
for layer in (2, 3, 4, 5):
    W[layer][:, :486] = 0
    B[layer][:486] = 0

# These global slots were unused at their respective layers.
W[2][:, 494:] = 0
B[2][494:] = 0
for layer in (3, 4, 5):
    W[layer][:, 496:] = 0
    B[layer][496:] = 0

# h2:
# channel 0: occupancy code, opponent=0, empty=1, own=2.
# channels 2/3: horizontal two-liberty certificates, own/opponent.
# channels 4/5: vertical two-liberty certificates, own/opponent.
#
# At an occupied center, direct liberties of the center and its
# opposite same-color neighbors are mutually disjoint.
for i in range(81):
    B[2][u(i, 0)] = 1
    add(2, u(i, 0), u(i, 0), 1)
    add(2, u(i, 1), u(i, 0), -1)

    for axis in (0, 1):
        for color in (0, 1):
            destination = u(i, 2 + 2*axis + color)
            B[2][destination] = -17
            add(2, u(i, color), destination, 16)
            add(2, u(i, 2 + color), destination, 1)
            for j in axes[i][axis]:
                add(2, u(j, 2 + color), destination, 1)

# Supplemental three-liberty certificates at a symmetric set of
# 49 sites. The center's direct liberties are disjoint from those
# of its neighbors; a neighbor liberty is counted at most twice.
spare2 = [u(i, 1) for i in range(81)] + list(range(494, 512))
full_sites = [
    i for i, (r, c) in enumerate(coords)
    if (r + c) % 2 == 0 or r == 4 or c == 4
]
assert len(full_sites) == 49

full_three = {}
slot = 0
for i in full_sites:
    for color in (0, 1):
        destination = spare2[slot]
        slot += 1
        full_three[(i, color)] = destination
        B[2][destination] = -18
        add(2, u(i, color), destination, 16)
        add(2, u(i, 2 + color), destination, 1)
        for j in neighbors[i]:
            add(2, u(j, 2 + color), destination, 0.5)
assert slot == 98

def gate(layer, point, color, destination, magnitude):
    # With occupancy code t:
    # own gate = G*(t-2), opponent gate = -G*t.
    if color == 0:
        B[layer][destination] = -2 * magnitude
        add(layer, u(point, 0), destination, magnitude)
    else:
        B[layer][destination] = 0
        add(layer, u(point, 0), destination, -magnitude)

# h3: merge and propagate both directional two-liberty proofs.
#
# If directional seeds are a=relu(Lh-1), b=relu(Lv-1),
# then a+b>2 implies that at least one directional bound is >=3.
# Supplemental neighboring three-liberty proofs are also merged.
for i in range(81):
    add(3, u(i, 0), u(i, 0), 1)
    for color in (0, 1):
        two = u(i, 2 + color)
        gate(3, i, color, two, 256)
        for j in [i] + neighbors[i]:
            add(3, u(j, 2 + color), two, 1)
            add(3, u(j, 4 + color), two, 1)

        three = u(i, 4 + color)
        gate(3, i, color, three, 256)
        B[3][three] -= np.float32(2)
        add(3, u(i, 2 + color), three, 1)
        add(3, u(i, 4 + color), three, 1)
        for j in [i] + neighbors[i]:
            source = full_three.get((j, color))
            if source is not None:
                # A positive supplemental proof is at least 1/2.
                add(3, source, three, 6)

# Additional exact directional three-liberty seeds.
# Retain the previous symmetry-equivariant forty-site selection,
# and cover both axes at four central diagonal sites.
directional = []
for i, (r, c) in enumerate(coords):
    if (r + c) % 2 == 1:
        directional.append((i, 0 if r % 2 == 0 else 1))
for r in (3, 5):
    for c in (3, 5):
        for axis in (0, 1):
            directional.append((9*r+c, axis))

spare3 = [u(i, 1) for i in range(81)] + list(range(496, 512))
extra_three = {}
slot = 0
for i, axis in directional:
    for color in (0, 1):
        destination = spare3[slot]
        slot += 1
        extra_three.setdefault((i, color), []).append(destination)
        add(3, u(i, 2 + 2*axis + color), destination, 1)
        B[3][destination] = -1
assert slot == 96

# h4: another same-color propagation step.
# Its otherwise unused channel 1 decodes own occupancy, enabling
# local same-color group-size estimates in h5.
proof_channels = [(2, 0), (3, 1), (4, 0), (5, 1)]
for i in range(81):
    add(4, u(i, 0), u(i, 0), 1)

    add(4, u(i, 0), u(i, 1), 1)
    B[4][u(i, 1)] = -1

    for channel, color in proof_channels:
        destination = u(i, channel)
        gate(4, i, color, destination, 2048)
        for j in [i] + neighbors[i]:
            add(4, u(j, channel), destination, 1)
            if channel in (4, 5):
                for source in extra_three.get((j, color), []):
                    add(4, source, destination, 1)

# h5 restores the scoring interface.
# Own threat flags remain binary so the existing rescue-liberty
# budget is unchanged. Opponent atari candidates receive a bounded
# size premium based on adjacent opposing stones.
for i in range(81):
    add(5, u(i, 0), u(i, 0), 1)
    B[5][u(i, 0)] = -1
    add(5, u(i, 0), u(i, 1), -1)
    B[5][u(i, 1)] = 1

    for channel, color in proof_channels:
        destination = u(i, channel)

        if channel == 3:
            # Opponent occupancy gate: 1 - 2*t.
            # Opp_j = decoded_own_j - t_j + 1.
            B[5][destination] = 1 + 0.25*len(neighbors[i])
            add(5, u(i, 0), destination, -2)
            for j in neighbors[i]:
                add(5, u(j, 1), destination, 0.25)
                add(5, u(j, 0), destination, -0.25)
            for j in [i] + neighbors[i]:
                add(5, u(j, channel), destination, -4)
        else:
            if color == 0:
                add(5, u(i, 0), destination, 1)
                B[5][destination] = -1
            else:
                add(5, u(i, 0), destination, -1)
                B[5][destination] = 1
            for j in [i] + neighbors[i]:
                add(5, u(j, channel), destination, -2)

# Distinguish diagonal false-eye patterns without removing the
# existing orthogonal enclosure and endangered-group exceptions.
for i, (r, c) in enumerate(coords):
    eye = u(i, 3)
    diagonal_weight = -0.5 if len(neighbors[i]) == 4 else -1.0
    for dr in (-1, 1):
        for dc in (-1, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 9 and 0 <= cc < 9:
                add(6, u(9*rr + cc, 1), eye, diagonal_weight)

for layer in range(2, 7):
    save("W" + str(layer), W[layer])
    save("b" + str(layer), B[layer])

print("Installed both-axis two-liberty certificates at all 81 points.")
print("Added 98 supplemental and 96 directional three-liberty seeds.")
print("Added bounded capture-size priority and diagonal eye corrections.")
