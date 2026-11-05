import numpy as np

W6 = load("W6")

def u(point, channel):
    return 6 * point + channel

EARLY = 486

for r in range(9):
    for c in range(9):
        i = 9*r + c
        base = u(i, 0)
        waste = u(i, 4)

        # Friendly contact supports connection and defense.
        # The existing concentration penalty still discourages
        # unnecessary connections inside uncontested own territory.
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 9 and 0 <= cc < 9:
                W6[u(9*rr + cc, 0), base] = np.float32(0.45)

        # Modest support for diagonal shape; one-space extensions
        # retain their existing, stronger positional coefficient.
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 9 and 0 <= cc < 9:
                    W6[u(9*rr + cc, 0), base] = np.float32(0.35)

        # Discourage cramped early development. This adjustment
        # fades with board density; nearby opposition already
        # relaxes the concentration gate through its existing weights.
        W6[EARLY, waste] = np.float32(0.65)

save("W6", W6)
print("Updated connection preferences and opening spacing.")
print("Preserved all liberty, capture, self-atari, eye, and pass circuits.")
