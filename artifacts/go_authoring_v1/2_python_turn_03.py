import numpy as np

W6 = load("W6")
b6 = load("b6")
W7 = load("W7")
b7 = load("b7")

def u(channel, point):
    return 81 * channel + point

SCORE_OFFSET = 64.0
FUSION_OFFSET = 16.0
RELIEF = 0.75

for q in range(81):
    base = u(0, q)
    rescue = u(1, q)
    capture = u(2, q)
    eye = u(4, q)

    gain = float(W7[base, q])
    capture_weight = float(W7[capture, q]) / gain
    eye_weight = float(W7[eye, q]) / gain

    # Fuse three formerly separate linear contributions into one neuron.
    # The eye feature is at most 2 on valid inputs; the added offset
    # therefore keeps this fused expression nonnegative.
    for target in (base, capture, eye):
        W6[:, target] = 0.0
        b6[target] = 0.0
        W7[target, :] = 0.0

    W6[base, base] = 1.0
    W6[capture, base] = capture_weight
    W6[eye, base] = eye_weight
    b6[base] = FUSION_OFFSET
    W7[base, q] = gain
    b7[q] -= np.float32(gain * FUSION_OFFSET)

    # Reuse the freed neurons for:
    #   D0 = relu(64 - strategic_score)
    #   D1 = relu(64 - strategic_score - 4*capture - 2*rescue)
    #
    # Their difference equals
    # min(relu(64 - strategic_score), 4*capture + 2*rescue).
    # Thus relief vanishes without tactical pressure, is bounded, and
    # never raises the strategic contribution above its neutral baseline.
    W6[base, capture] = -1.0
    b6[capture] = SCORE_OFFSET

    W6[base, eye] = -1.0
    W6[capture, eye] = -4.0
    W6[rescue, eye] = -2.0
    b6[eye] = SCORE_OFFSET

    W7[capture, q] = gain * RELIEF
    W7[eye, q] = -gain * RELIEF

# Existing filtered rescue, self-atari, contact, and passing circuits
# remain unchanged.
save("W6", W6)
save("b6", b6)
save("W7", W7)
save("b7", b7)

print("Added bounded tactical relief using two freed spatial channels.")
print("No training, search, simulation, or neural evaluations performed.")
