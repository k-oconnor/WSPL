"""Explicit reconstructed DDPM sampler for the released 12x12 x0 predictors.

The release specifies the schedule, but not sampler code or RNG. This is a
declared posterior-variance implementation, not bitwise reproduction.
"""
import hashlib
from pathlib import Path
import numpy as np

SHAPES = {'W1': (160, 64), 'b1': (64,), 'W2': (64, 64),
          'b2': (64,), 'W3': (64, 144), 'b3': (144,)}
MODEL_IDS = ('246a5b58', 'a6a9dfb7', 'd88aba69')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_embedding(t):
    if not isinstance(t, (int, np.integer)) or not 0 <= t < 1000:
        raise ValueError('integer time in [0,999] required')
    frequencies = 10000. ** (-np.arange(8, dtype=np.float64) / 8.)
    return np.concatenate((np.sin(t * frequencies), np.cos(t * frequencies))).astype(np.float32)


class Denoiser:
    def __init__(self, arrays):
        if set(arrays) != set(SHAPES):
            raise ValueError('exact six-tensor layout required')
        self.weights = {}
        for name, shape in SHAPES.items():
            a = np.asarray(arrays[name])
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError('invalid tensor: ' + name)
            self.weights[name] = a.copy()
            self.weights[name].flags.writeable = False

    def predict(self, pixels, t):
        pixels = np.asarray(pixels, dtype=np.float32)
        if pixels.ndim != 2 or pixels.shape[1] != 144 or not np.isfinite(pixels).all():
            raise ValueError('finite batch of 144 pixels required')
        h = np.concatenate((pixels, np.broadcast_to(time_embedding(t), (len(pixels), 16))), axis=1)
        for i in range(1, 4):
            h = h @ self.weights['W' + str(i)] + self.weights['b' + str(i)]
            if i < 3:
                h = np.maximum(h, np.float32(0))
        if not np.isfinite(h).all():
            raise ValueError('nonfinite prediction')
        return h


class PosteriorSampler:
    def __init__(self):
        # Coefficients computed in float64 then cast once for FP32 transitions.
        beta = np.linspace(.0001, .02, 1000, dtype=np.float64)
        alpha = 1. - beta
        abar = np.cumprod(alpha)
        prev = np.concatenate(([1.], abar[:-1]))
        self.beta, self.abar = beta, abar
        self.clean = (beta * np.sqrt(prev) / (1. - abar)).astype(np.float32)
        self.noisy = (np.sqrt(alpha) * (1. - prev) / (1. - abar)).astype(np.float32)
        self.variance = (beta * (1. - prev) / (1. - abar)).astype(np.float32)
        self.clean[0], self.noisy[0], self.variance[0] = 1., 0., 0.

    def step(self, pixels, prediction, t, noise=None):
        time_embedding(t)  # Validate the index, including negative indices.
        x = np.asarray(pixels, dtype=np.float32)
        p = np.asarray(prediction, dtype=np.float32)
        if x.shape != p.shape or x.ndim != 2 or x.shape[1] != 144:
            raise ValueError('matching batched pixels required')
        if not np.isfinite(x).all() or not np.isfinite(p).all():
            raise ValueError('finite inputs required')
        clean = np.clip(p, -1., 1.)
        if t == 0:
            if noise is not None:
                raise ValueError('final step must not receive noise')
            return clean
        z = np.asarray(noise, dtype=np.float32)
        if z.shape != x.shape or not np.isfinite(z).all():
            raise ValueError('finite noise of matching shape required')
        result = self.clean[t] * clean + self.noisy[t] * x + np.sqrt(self.variance[t]) * z
        if not np.isfinite(result).all():
            raise ValueError('nonfinite transition')
        return result

    def sample(self, model, count, seed, callback=None):
        if not isinstance(count, int) or not 1 <= count <= 64:
            raise ValueError('batch size must be 1..64')
        rng = np.random.Generator(np.random.PCG64(seed))
        # Explicit float64 normal draw -> FP32 conversion, also for initial noise.
        x = rng.standard_normal((count, 144)).astype(np.float32)
        traces = {}
        for t in range(999, -1, -1):
            if callback is not None:
                callback(t)
            if t in (999, 749, 499, 249, 0):
                traces['x_t_' + str(t)] = x[:1].copy()
            prediction = model.predict(x, t)
            noise = rng.standard_normal(x.shape).astype(np.float32) if t else None
            x = self.step(x, prediction, t, noise)
        traces['final'] = x[:1].copy()
        return x, traces


class DiffusionReplayStudy:
    def __init__(self, reference_root, model_hashes):
        if set(model_hashes) != set(MODEL_IDS):
            raise ValueError('three pinned models required')
        self.models = {}
        for name in MODEL_IDS:
            path = Path(reference_root) / (name + '.npz')
            if sha(path) != model_hashes[name]:
                raise ValueError('model identity differs: ' + name)
            with np.load(path, allow_pickle=False) as z:
                self.models[name] = Denoiser({k: z[k] for k in z.files})
        self.sampler = PosteriorSampler()


def contact_sheet_pgm(samples, path):
    """Lossless dependency-free 8x8 preview; all quantitative data stay in NPZ."""
    x = np.asarray(samples)
    if x.shape != (64, 144) or not np.isfinite(x).all():
        raise ValueError('64 finite samples required')
    pixels = np.rint((np.clip(x, -1., 1.) + 1.) * 127.5).astype(np.uint8)
    sheet = pixels.reshape(8, 8, 12, 12).transpose(0, 2, 1, 3).reshape(96, 96)
    Path(path).write_bytes(b'P5\n96 96\n255\n' + sheet.tobytes())
