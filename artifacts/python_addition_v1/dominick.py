_LEVELS = {
    "0": 0, "a": 5171, "b": 5428, "c": 5227, "d": 5493,
    "e": 974, "f": 2447, "g": 3477, "h": 3496, "i": 5967,
    "j": 5740, "k": 2946, "l": 2349, "m": 2748, "n": 6153,
    "o": 8547, "p": 3736, "q": 4158, "r": 1796, "s": 1761,
    "t": 1766, "u": 2317, "v": 2319, "w": 2321, "z": 3270,
    "A": 3335, "B": 4636, "C": 4955, "D": 5427, "E": 5221,
    "F": 5250, "G": 5277, "H": 5543
}

_SEQUENCES = """
aaaa0a0a0
a0a0a00aa
gggggghhh
0fff000ff
0vunnnzgg
ff0fff000
dd0eeeeee
a0aaaa0a0
a0jjj00H0
nnzgggggg
gghhhh0hh
bbbbbbbbb
ccccccccd
kkk0k0kkk
ccccccddd
bbbbbbb0b
aaaaa0aaa
eeeeee000
0ii0iiiii
ooC0p00pp
hhhh0hhhh
eeeee000e
mm0tswwv0
aaajj00a0
0DDE0G0Fc
zgggggggg
bbbbbbbbb
ccddd0eee
aa0aaaa0a
b00DDE0G0
i00i0ii0i
bbb0bbbbb
j00a0jjj0
aa0a0a00a
o0ooC0p00
gggghhhh0
v0vunnnzg
kkkrlllll
rlllllk0q
ddd0eeeee
aajj00a0j
bbb0bb00D
0ooC0p00p
f000ff00f
unnnzgggg
ii0iiiiii
00ff00fo0
bbbb0bb00
pA00kkk0k
0ppppp0pA
eeeeeeeee
hhhBii0i0
f0fff0fff
Bii0i00i0
00aaajj00
A00kkk0k0
b0bb00DDE
lk0qqq0qq
aaaa0aaaa
krlllllk0
f0fff000f
pp0pA00kk
bbbb0bbbb
DDE0G0Fcc
fo0ooC0p0
bb0bbbbbb
0fff0fff0
bbbbbb0bb
0G0Fccccc
b0bbbbbbb
jj00H0HHb
cccccccdd
swwv0vunn
Fcccccccc
a0a00aaaj
bbbbb0bb0
000ee0ff0
0aaaaa0aa
ggggghhhh
qmmm0tsww
d0eeeeeee
0hhhhBii0
lllk0qqq0
0bbbbbbbb
G0Fcccccc
j00H0HHbb
0i00i0ii0
llk0qqq0q
00fo0ooC0
f00fo0ooC
0fo0ooC0p
wwv0vunnn
0pA00kkk0
DE0G0Fccc
C0p00pppp
0kkk0k0kk
eeeeeeee0
eee000ee0
0kkk0rlll
hhhhBii0i
ppppp0pA0
hh0hhhhBi
mmm0tswwv
00ee0ff0f
ajj00a0jj
cccddd0ee
kk0k0kkkk
ppp0pA00k
0ee0ff0ff
0aaajj00a
aaa0aaaa0
ghhhh0hhh
00a0jjj00
aaa0a0a00
000ff00fo
eeee000ee
qqq0qqqmm
m0tswwv0v
eeeeeee00
cddd0eeee
a00aaaaa0
ee0ff0fff
e000ee0ff
ggggggggh
0HHbbbb0b
E0G0Fcccc
ggghhhh0h
ff0fff0ff
""".split()

_EXAMPLES = np.array(
    [[_LEVELS[c] for c in row] for row in _SEQUENCES],
    dtype=np.int64
)

_MEAN = np.array([
    32.51843725144863, 31.15882784128189,
    31.0840622484684, 30.86609348654747,
    31.181484162807465, 31.356796622276306,
    29.210468500852585, 31.39812469482422
], dtype=np.float64)

_STD = np.array([
    21.9948662276044, 23.10261831552019,
    22.91111776877668, 23.014046392648023,
    23.381239809194867, 22.81378748787699,
    23.584313494548567, 22.540201691644427
], dtype=np.float64)

_RX = _EXAMPLES[:, :8].astype(np.float64) * 0.01
_RY = _EXAMPLES[:, 8].astype(np.float64) * 0.01
_RPOS = _RX > 0.005
_RINDEX = np.max(
    np.where(_RPOS, np.arange(8)[None, :], -1), axis=1
)
_RLAST = _RX[np.arange(len(_RX)), np.maximum(_RINDEX, 0)]
_RAGE = 7 - _RINDEX
_RAVAILABLE = (_RY > 0.005).astype(np.float64)
_RCHANGE = (_RY - _RLAST) * _RAVAILABLE

_TIME_WEIGHTS = np.array(
    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    dtype=np.float64
)
_TIME_TOTAL = float(np.sum(_TIME_WEIGHTS))
_CONTEXT_BLEND = {4: 0.18, 5: 0.32, 6: 0.47, 7: 0.62, 8: 0.72}


def predict(x):
    a = np.asarray(x, dtype=np.float64)
    out = np.empty((a.shape[0], 1), dtype=np.float32)

    for start in range(0, a.shape[0], 256):
        stop = min(start + 256, a.shape[0])
        raw = a[start:stop] * _STD + _MEAN
        raw = np.nan_to_num(
            raw, nan=0.0, posinf=1000000.0, neginf=0.0
        )
        cents = np.rint(
            np.clip(raw, 0.0, 1000000.0) * 100.0
        ).astype(np.int64)
        values = cents.astype(np.float64) * 0.01
        positive = cents > 0
        n = len(values)

        last_index = np.max(
            np.where(positive, np.arange(8)[None, :], -1), axis=1
        )
        last = values[np.arange(n), np.maximum(last_index, 0)]
        last = np.where(last_index >= 0, last, 0.0)
        age = 7 - last_index

        observed_fraction = (
            np.sum(positive * _TIME_WEIGHTS, axis=1) / _TIME_TOTAL
        )
        prior_availability = 0.86 - 0.60 * (1.0 - observed_fraction)

        # Compare nonzero levels separately from zero-observation patterns.
        both = positive[:, None, :] & _RPOS[None, :, :]
        level_scale = 0.75 + 0.025 * last
        difference = (
            values[:, None, :] - _RX[None, :, :]
        ) / level_scale[:, None, None]
        history_distance = np.sum(
            np.minimum(difference * difference, 16.0)
            * both * _TIME_WEIGHTS,
            axis=2
        ) / np.maximum(
            np.sum(both * _TIME_WEIGHTS, axis=2), 1.0
        )

        mask_distance = np.sum(
            (positive[:, None, :] != _RPOS[None, :, :])
            * _TIME_WEIGHTS,
            axis=2
        ) / _TIME_TOTAL

        current_scale = 0.18 + 0.0015 * last
        current_distance = (
            (last[:, None] - _RLAST[None, :])
            / current_scale[:, None]
        ) ** 2

        distance = (
            current_distance
            + 0.60 * history_distance
            + 2.50 * mask_distance
            + 0.30 * (age[:, None] - _RAGE[None, :]) ** 2
        )
        weights = np.exp(-0.5 * np.minimum(distance, 150.0))
        total = np.sum(weights, axis=1)

        availability = (
            5.0 * prior_availability
            + np.sum(weights * _RAVAILABLE[None, :], axis=1)
        ) / (5.0 + total)

        positive_weight = np.sum(
            weights * _RAVAILABLE[None, :], axis=1
        )
        change = np.sum(
            weights * _RCHANGE[None, :], axis=1
        ) / (5.0 + positive_weight)
        level = np.maximum(last + change, 0.0)
        forecast = availability * level

        # Short all-zero suffixes do not identify a price regime.
        # Only substantial contexts containing positive observations
        # can override the level-preserving forecast.
        for length in range(4, 9):
            suffix = cents[:, -length:]
            eligible = np.sum(suffix > 0, axis=1) >= 2
            matches = np.all(
                suffix[:, None, :] == _EXAMPLES[None, :, :length],
                axis=2
            )
            matches &= eligible[:, None]
            count = np.sum(matches, axis=1)

            continuation = np.sum(
                matches * (_EXAMPLES[None, :, length] * 0.01),
                axis=1
            ) / np.maximum(count, 1)

            minimum = np.min(
                np.where(suffix > 0, suffix, 1000000000), axis=1
            )
            maximum = np.max(suffix, axis=1)
            varied = maximum - minimum > 1
            blend = _CONTEXT_BLEND[length] * np.where(varied, 1.0, 0.65)

            # Replace the context component rather than repeatedly
            # accumulating overlapping evidence.
            candidate = (
                (1.0 - blend) * availability * level
                + blend * continuation
            )
            forecast = np.where(count > 0, candidate, forecast)

        # Long unobserved runs should not inherit unrelated nonzero
        # continuations from the reference examples.
        forecast *= 0.70 ** np.maximum(age - 3, 0)
        forecast = np.where(last_index >= 0, forecast, 0.0)

        standardized = (
            forecast - 29.171249717473984
        ) / 22.607836247690592
        out[start:stop, 0] = np.nan_to_num(
            standardized, nan=-1.290315866470337,
            posinf=100000.0, neginf=-1.290315866470337
        ).astype(np.float32)

    return out