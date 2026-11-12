def predict(x):
    a = np.asarray(x)
    z = np.nan_to_num(
        np.asarray(a[:, [10, 11, 13, 14, 15, 26, 28, 177]],
                   dtype=np.float64),
        nan=0.0, posinf=12.0, neginf=-12.0
    )

    black = ((z[:, 0] > 0.1) & (z[:, 0] < 1.3)).astype(np.float64)
    hispanic = ((z[:, 1] > -0.1) & (z[:, 1] < 3.7)).astype(np.float64)
    unmarried = (z[:, 2] > 0.0).astype(np.float64)
    education = np.where(
        z[:, 3] > 2.4, 0.0, np.clip(z[:, 3], -1.75, 2.12)
    )
    father_unknown = (z[:, 4] > 1.5).astype(np.float64)
    gain = np.clip(z[:, 5], -1.6, 2.561684)
    gain_unknown = (z[:, 5] > 2.4).astype(np.float64)
    cigarettes = np.clip(7.78125 + 25.28893579092446 * z[:, 6], 0.0, 99.0)
    treatment = np.clip(z[:, 7], 0.0, 1.0)

    smoking_component = np.where(
        cigarettes > 90.0, 2.376, 0.012 * cigarettes
    )

    baseline = (
        -0.36
        + smoking_component
        + 0.48 * black
        + 0.04 * education
        + 0.06 * gain
        - 0.05 * (hispanic - 0.234375)
        - 0.05 * (father_unknown - 0.171875)
    )

    effect = (
        0.135
        + 0.18 * (hispanic - 0.234375)
        - 0.08 * education
        - 0.14 * gain
        - 0.10 * (gain_unknown - 0.1015625)
        + 0.025 * (unmarried - 0.4453125)
    )

    y = baseline + treatment * effect
    return np.clip(y, -20.0, 20.0).astype(np.float32).reshape(-1, 1)