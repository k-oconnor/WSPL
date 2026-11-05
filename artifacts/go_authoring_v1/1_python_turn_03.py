import numpy as np

W6, b6 = load("W6"), load("b6")
W7 = load("W7")

def ch(channel, point):
    return 81 * channel + point

scale = np.float32(1.20)

for r in range(9):
    for c in range(9):
        s = 9*r + c
        neighbors = [
            9*rr + cc
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
            if 0 <= rr < 9 and 0 <= cc < 9
        ]

        # H5 channels:
        # 0,1: own/opponent occupancy
        # 2,3: own/opponent lacking a two-liberty certificate
        # 4,5: own/opponent lacking a three-liberty certificate
        #
        # Fold nonnegative tactical terms into the shifted affine
        # shape channel. Its existing offset remains sufficient:
        # adding nonnegative terms cannot activate its lower clipping.
        shape_dest = ch(0, s)
        for t in neighbors:
            W6[ch(2, t), shape_dest] += np.float32(6.00)
            W6[ch(4, t), shape_dest] += np.float32(1.50)
            W6[ch(3, t), shape_dest] += np.float32(9.90)
            W6[ch(5, t), shape_dest] += np.float32(2.60)

        # Replace stone-wise urgency with a candidate-wise discount.
        # C counts neighboring stones lacking two-liberty certificates.
        # The discount is 6*relu(C-1), shared across both colors.
        #
        # Thus additional rescue contacts have much less influence,
        # while additional capture contacts retain a larger reward.
        # This is a contact heuristic, not an exact group-identity test.
        discount_dest = ch(2, s)
        W6[:, discount_dest] = 0
        b6[discount_dest] = -1

        for t in neighbors:
            W6[ch(2, t), discount_dest] = 1
            W6[ch(3, t), discount_dest] = 1

        # Remove the old neighboring-stone urgency readout.
        W7[discount_dest, :] = 0
        W7[discount_dest, s] = np.float32(-6.0 * scale)

# Liberty proofs, self-atari checks, eye protection, enclosure,
# opening preferences, and passing remain unchanged.
for name, array in (("W6", W6), ("b6", b6), ("W7", W7)):
    assert np.isfinite(array).all()
    save(name, array)

print("Added diminishing returns for repeated atari contacts.")
print("Construction used coefficient edits only; no policy evaluation.")
