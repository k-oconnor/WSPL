def predict(x):
    a = np.asarray(x, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=20.0, neginf=-20.0)
    b = a[:, :25] > 0.0

    s = (
        1.355
        + 0.188443203 * np.clip(a[:, 2], -15.0, 15.0)
        + 0.387556037 * np.clip(a[:, 4], -15.0, 15.0)
        + 0.10 * b[:, 6]
        + 0.40 * b[:, 8]
        + 0.10 * b[:, 10]
        + 0.40 * b[:, 12]
        + 0.40 * b[:, 13]
        + 0.05 * b[:, 14]
        + 0.30 * b[:, 16]
        + 0.40 * b[:, 23]
        + 0.20 * b[:, 24]
    )

    control = (
        np.exp(np.clip(s, -12.0, 8.0)) - 11.065397688177757
    ) / 5.358131601443741

    treated = 0.9198 + (s - 2.277) / 5.358131601443741

    result = np.where(a[:, 25] > 0.5, treated, control)
    return result.astype(np.float32).reshape(-1, 1)