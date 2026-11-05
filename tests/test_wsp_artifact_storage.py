"""Bulk artifact storage must match the original element-wise implementation exactly."""
import json
import re
import unittest

import numpy as np

from skillforge.wsp_artifact import TensorArtifact, Bits, canonical, diff, hex_words, FORMAT


def reference_text(arrays, graph):
    """Original from_arrays serialization (one f-string per element)."""
    tensors = {k: dict(shape=list(a.shape), dtype='f32', bits=[f'{int(b):08x}' for b in np.ascontiguousarray(a).view(np.uint32).flat])
               for k, a in arrays.items()}
    return canonical(dict(format=FORMAT, graph=graph, tensors=tensors))


def reference_words_error(words):
    """Original per-word validation loop; returns its error message or None."""
    try:
        for word in words:
            if not isinstance(word, str) or not re.fullmatch('[0-9a-f]{8}', word):
                raise ValueError('canonical eight-digit FP32 hex bits required')
            Bits(int(word, 16))
    except ValueError as e:
        return str(e)
    return None


def reference_diff(a, b, offset, limit):
    """Original string-by-string diff loop (changes and counts only)."""
    a, b = a.to_dict(), b.to_dict()
    changed = []; count = 0; tensors = {}
    for name in sorted(a['tensors']):
        ar, br = a['tensors'][name], b['tensors'][name]
        local = 0
        for i, (old, new) in enumerate(zip(ar['bits'], br['bits'])):
            if old == new:
                continue
            if offset <= count < offset + limit:
                changed.append(dict(tensor=name, flat_index=i, index=list(map(int, np.unravel_index(i, ar['shape']))), before_bits=old, after_bits=new))
            count += 1; local += 1
        if local:
            tensors[name] = local
    return count, tensors, changed


def random_bits(rng, shape):
    bits = rng.integers(0, 2**32, size=shape, dtype=np.uint64).astype(np.uint32)
    bits[(bits & 0x7f800000) == 0x7f800000] &= 0xbfffffff  # Keep finite.
    flat = bits.reshape(-1)
    flat[::7] = 0x80000000; flat[::11] = 0; flat[::13] &= 0x807fffff  # Signed zeros, subnormals.
    return bits


class StorageTests(unittest.TestCase):
    def test_serialization_and_identity_unchanged(self):
        rng = np.random.default_rng(3)
        graph = dict(kind='opaque', label='uninterpreted tensor bank')
        for shapes in ({'A': (3, 5), 'b': (5,)}, {'z': (1,), 'M': (2, 3, 4)}):
            arrays = {k: random_bits(rng, s).view(np.float32) for k, s in shapes.items()}
            artifact = TensorArtifact.from_arrays(arrays)
            self.assertEqual(artifact._text, reference_text(arrays, graph))
            reloaded = TensorArtifact(json.loads(artifact._text))
            self.assertEqual(reloaded._text, artifact._text)
            self.assertEqual(reloaded.sha256, artifact.sha256)
            for k, a in reloaded.arrays(bits=True).items():
                np.testing.assert_array_equal(a, arrays[k].view(np.uint32))
            # Returned arrays are copies; cached storage is read-only.
            reloaded.arrays(bits=True)['A' if 'A' in shapes else 'z'][...] = 1
            self.assertEqual(TensorArtifact(json.loads(artifact._text)).sha256, reloaded.sha256)
            with self.assertRaises(ValueError):
                next(iter(reloaded._words.values()))[...] = 1

    def test_with_bits_matches_rebuild(self):
        rng = np.random.default_rng(5)
        parent = TensorArtifact.dense([4, 3, 2], ['relu', 'linear'])
        arrays = {k: random_bits(rng, a.shape) for k, a in parent.arrays(bits=True).items()}
        fast = parent.with_bits(arrays)
        d = parent.to_dict()
        for k, a in arrays.items():
            d['tensors'][k]['bits'] = [f'{int(b):08x}' for b in a.flat]
        self.assertEqual(fast._text, TensorArtifact(d)._text)
        self.assertEqual(parent.to_dict(), TensorArtifact.dense([4, 3, 2], ['relu', 'linear']).to_dict())
        bad = dict(arrays, W1=arrays['W1'].copy()); bad['W1'][0, 0] = 0x7f800000
        with self.assertRaisesRegex(ValueError, 'nonfinite FP32 bits are unsupported'):
            parent.with_bits(bad)

    def test_malformed_words_raise_original_errors(self):
        good = ['3f800000', '00000000', '80000000']
        cases = [good, ['3F800000'] + good[1:], ['3f80000'] + good[1:], good[:2] + ['3f8000000'],
                 [1] + good[1:], good[:1] + ['7f800000'] + good[2:], ['ff800001'] + good[1:],
                 good[:2] + ['7fc00000'], ['3f80000g'] + good[1:], ['0000000', '00000000', '0']]
        for words in cases:
            with self.subTest(words=words):
                document = dict(format=FORMAT, graph=dict(kind='opaque', label='x'),
                                tensors={'A': dict(shape=[3], dtype='f32', bits=list(words))})
                expected = reference_words_error(words)
                if expected is None:
                    self.assertEqual(TensorArtifact(document).arrays(bits=True)['A'].tolist(), [int(w, 16) for w in words])
                else:
                    with self.assertRaises(ValueError) as caught:
                        TensorArtifact(document)
                    self.assertEqual(str(caught.exception), expected)

    def test_element_limit_and_within(self):
        artifact = TensorArtifact.from_arrays({'A': np.zeros((4, 4), np.float32)})
        self.assertIs(artifact.within(16), artifact)
        with self.assertRaisesRegex(ValueError, 'tensor element limit exceeded'):
            artifact.within(15)
        with self.assertRaisesRegex(ValueError, 'tensor element limit exceeded'):
            TensorArtifact(artifact.to_dict(), max_elements=15)

    def test_diff_pages_match_reference(self):
        rng = np.random.default_rng(11)
        parent = TensorArtifact.from_arrays({k: random_bits(rng, s).view(np.float32) for k, s in {'W': (9, 7), 'a': (5,), 'b': (300,)}.items()})
        arrays = parent.arrays(bits=True)
        for k, a in arrays.items():
            mask = rng.random(a.shape) < .3
            a[mask] = random_bits(rng, a.shape)[mask]
        candidate = parent.with_bits(arrays)
        for offset in (0, 1, 5, 17, 40, 200):
            for limit in (0, 1, 3, 64, 4096):
                result = diff(parent, candidate, offset=offset, limit=limit)
                count, tensors, changes = reference_diff(parent, candidate, offset, limit)
                self.assertEqual((result['changed_elements'], result['changed_by_tensor'], result['changes']), (count, tensors, changes))

    def test_hex_words(self):
        words = np.array([0, 1, 0x80000000, 0x3f800000, 0x7f7fffff], np.uint32)
        self.assertEqual(hex_words(words), [f'{int(w):08x}' for w in words])


if __name__ == '__main__':
    unittest.main()
