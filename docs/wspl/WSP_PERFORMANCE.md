# WSP editor performance

November 5, 2026. This is an engineering change to the WSPL 0.2 editor (issue #10). It changes no language semantics, no study protocol and no pinned isolated checkout.

## Acceptance rule: byte identity

An optimization is accepted only if every archived WSPL program gives the same result as on `main` a9d7510:

- the same candidate `sha256`;
- the same plan body apart from `implementation_id`, which covers:
  - writes and flat-index order;
  - the rounding count and the first 64 examples;
  - charged `resources.operations`;
  - `touched` and `diff`;
- the same error text for failing programs.

`implementation_id` hashes the editor source, so it changes with any edit. Artifact and plan formats, hashes, limits and charged totals are all unchanged.

To reproduce:

```sh
python tools/wsp_editor_oracle.py --check tests/fixtures/wsp_editor_oracle_v1.json   # 73 cases, incl. 40 Go replays
python tools/bench_wsp_editor.py                                                    # seeded 1.48M-parameter Go parent
WSPL_ORACLE_FULL=1 python -m pytest tests/test_wsp_editor_oracle.py
```

The fixture was generated on unmodified `main`, where it also reproduces the archived Breakout and repair candidate IDs. Breakout and repair are the cases whose parents are in Git.

## What changed

1. **Artifact storage.**
   - Each `TensorArtifact` now also caches its validated `uint32` words, graph, hash and parsed document. Its canonical text and hash are unchanged.
   - Hex words are validated in bulk. If a word is malformed, the original per-word loop runs instead, so error messages are unchanged.
   - `diff` now compares the cached word arrays directly.
2. **Editor caching.**
   - Plans reuse the immutable parent.
   - `implementation_id` is cached.
   - Each construction statement's text is parsed once, and each literal is parsed once.
3. **Writes.**
   - Each distinct right-hand-side value is rounded once, and the results are broadcast.
   - `+=` and `-=` use float64 wherever the TwoSum error is zero, which means the float64 sum is exact. `*=` uses float64 when the right-hand side is FP32-representable, since an FP32×FP32 product is exact in float64.
   - Exact-zero results are stored as +0, matching the `Fraction` path.
   - Anything else falls back to the original element-wise exact path. That covers:
     - non-dyadic right-hand sides for augmented ops;
     - overflow;
     - precision limits below 1,100 bits;
     - inexact sums;
     - non-rational values.
   - The fallback raises any error itself, in the original order.
4. **Scalar arithmetic.** 0-d `+ - * /` and unary operations compute on the two scalars directly. They charge the same single operation as before.

## Timings

Measured on Python 3.11.16 and NumPy 1.26.4 on Linux, with the seeded synthetic Go parent. Times are seconds for `plan` plus `apply`.

| Program | `main` | Storage and caching | Plus writes and scalar arithmetic |
| --- | ---: | ---: | ---: |
| `go_authoring_v1/0_wspl_turn_00` (33,527 writes) | 24.3 | 17.3 | **6.0** |
| `go_authoring_v1/2_wspl_turn_00` (15,887 writes) | 14.8 | 9.8 | **2.8** |
| `go_authoring_v1/2_wspl_turn_06` (320 writes) | 4.9 | 0.8 | **0.7** |
| All 40 Go oracle replays | 406 | 230 | **104** |
| 9 diffusion oracle replays | 1.9 | 1.1 | **0.9** |

`apply` still replays the whole plan to check it, so it takes about as long as `plan`. Most of the remaining time goes to:

- evaluating index expressions with exact `Fraction` arithmetic;
- serializing plans of about 50 MB, which embed the full parent and candidate and every written index.

A smaller plan format (`wsp.edit_plan.v2`) would remove most of the serialization cost. It is deferred because it changes the format.
