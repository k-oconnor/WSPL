import numpy as np

W6 = load("W6")
b6 = load("b6")
W7 = load("W7")

CAPTURE_ACTIVITY = 495
CAPTURE_EXCESS = 496  # Replaces the obsolete raw-rescue aggregate.
CAPTURE_CAP = 12.0

# Together these neurons implement min(total_capture_pressure, 12).
W6[:, CAPTURE_EXCESS] = W6[:, CAPTURE_ACTIVITY].copy()
b6[CAPTURE_EXCESS] = b6[CAPTURE_ACTIVITY] - CAPTURE_CAP

W7[CAPTURE_EXCESS, :] = 0.0
W7[CAPTURE_EXCESS, PASS] = -W7[CAPTURE_ACTIVITY, PASS]

# Use the existing per-point rescue features after their self-atari
# and eye-filling filters. No additional hidden layer is needed.
gain = float(W7[0, 0])
for q in range(81):
    filtered_rescue = 81 + q
    W7[filtered_rescue, PASS] = np.float32(-0.08 * gain)

save("W6", W6)
save("b6", b6)
save("W7", W7)

print("Refined passing aggregates; placement logits unchanged.")
print("Coefficient construction only; no neural evaluations performed.")
