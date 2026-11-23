"""Data-only WSPL editing of LoRA factors. No author calls or training entry point.

Each projection's A/B pair is a small opaque WSPL tensor bank. A whole-adapter
identity binds the shards together; no graph, base weight, rank or scaling change
is admitted. Both native replacements and WSPL programs use the same final
shape/dtype/finiteness checks. Construction receipts are not behavioral proofs.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import re

import numpy as np

from .wsp_artifact import TensorArtifact
from .wsp_editor import plan, apply
from .wsp_syntax import parse
from .wsp_values import Limits


KEY = re.compile(r'([A-Za-z_][A-Za-z_0-9.]*)\.lora_([AB])\Z')


@dataclass(frozen=True)
class AdapterLimits:
    max_adapter_elements: int = 25_000_000
    max_pair_elements: int = 1_000_000
    max_program_bytes: int = 1_000_000
    max_pairs: int = 256
    max_total_operations: int = 64_000_000
    max_total_statements: int = 100_000
    max_total_loop_iterations: int = 100_000

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('Positive integer adapter limits required')


def snapshot(state, limits=AdapterLimits()):
    """Validate and copy an explicitly FP32, independent LoRA-only state."""
    if not isinstance(state, dict) or not state:
        raise ValueError('Nonempty LoRA state required')
    pairs = {}; total = 0; arrays = []
    for name, value in sorted(state.items()):
        match = KEY.fullmatch(name) if isinstance(name, str) else None
        if not match:
            raise ValueError('Only named lora_A/lora_B factors are allowed')
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype('float32'):
            raise ValueError('Explicit FP32 factors required; no silent conversion')
        if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
            raise ValueError('Finite nonempty rank-two factors required')
        if any(np.shares_memory(value, other) for other in arrays):
            raise ValueError('Shared factor storage is unsupported')
        arrays.append(value); total += value.size
        pairs.setdefault(match[1], {})[match[2]] = value
    if total > limits.max_adapter_elements or len(pairs) > limits.max_pairs:
        raise ValueError('Adapter capacity limit exceeded')
    for factors in pairs.values():
        if set(factors) != {'A', 'B'} or factors['A'].shape[0] != factors['B'].shape[1]:
            raise ValueError('Complete A/B pair with matching rank required')
        if sum(a.size for a in factors.values()) > limits.max_pair_elements:
            raise ValueError('Pair capacity limit exceeded')
    return {name: np.array(value, dtype=np.float32, order='C', copy=True)
            for name, value in sorted(state.items())}


def identity(state, limits=AdapterLimits()):
    checked = snapshot(state, limits)
    h = hashlib.sha256(b'wspl.lora.state.v1\0')
    for name, array in checked.items():
        metadata = json.dumps([name, list(array.shape), 'f32-le'], separators=(',', ':')).encode()
        h.update(len(metadata).to_bytes(8, 'little')); h.update(metadata)
        h.update(array.astype('<f4', copy=False).tobytes(order='C'))
    return h.hexdigest()


def _pair(state, projection, limits):
    keys = {factor: f'{projection}.lora_{factor}' for factor in ('A', 'B')}
    if not all(key in state for key in keys.values()):
        raise ValueError('Unknown projection')
    return TensorArtifact.from_arrays({factor: state[key].copy() for factor, key in keys.items()},
                                      max_elements=limits.max_pair_elements)


def describe(state, limits=AdapterLimits()):
    checked = snapshot(state, limits)
    projections = sorted({KEY.fullmatch(name)[1] for name in checked})
    return dict(format='wspl.lora.catalog.v1', adapter_id=identity(checked, limits),
                element_count=sum(a.size for a in checked.values()),
                projections={p: dict(pair_id=_pair(checked, p, limits).sha256,
                    A_shape=list(checked[p+'.lora_A'].shape),
                    B_shape=list(checked[p+'.lora_B'].shape)) for p in projections})


def source_for_pair(pair_id, body):
    if not re.fullmatch('[0-9a-f]{64}', pair_id):
        raise ValueError('Exact pair SHA-256 required')
    return ('wspl 0.2;\n'
            f'target adapter=artifact("lora_pair",sha256="{pair_id}");\n'
            'edit adapter as candidate mode=fixed numeric=exact_to_storage {\n'
            + body + '\n}\n')


def _prepare(parent, expected_parent_id, limits):
    original = snapshot(parent, limits)
    if identity(original, limits) != expected_parent_id:
        raise ValueError('Stale whole-adapter parent identity')
    return original, snapshot(original, limits)


def _finish(original, candidate, edits, arm, limits):
    candidate = snapshot(candidate, limits)
    if original.keys() != candidate.keys():
        raise ValueError('Adapter topology cannot change')
    changed = {}
    for name, before in original.items():
        after = candidate[name]
        if before.shape != after.shape:
            raise ValueError('Adapter rank/shape cannot change')
        count = int(np.count_nonzero(before.view(np.uint32) != after.view(np.uint32)))
        if count:
            changed[name] = count
    return candidate, dict(format='wspl.lora.construction.v1', arm=arm,
        parent_id=identity(original, limits), candidate_id=identity(candidate, limits),
        changed_elements=sum(changed.values()), changed_tensors=changed, edits=edits,
        behavioral_claim='none', base_weights='not accepted by this bridge',
        rank_and_scale='rank preserved; scale is external frozen runtime metadata')


def construct_wspl(parent, programs, *, expected_parent_id,
                   adapter_limits=AdapterLimits(), editor_limits=Limits()):
    """Atomically replay one program per selected pair against the same parent.

    Bindings and imported modules are intentionally absent. Only the archived
    parent factors, literals and bounded construction arithmetic are available.
    All proposals stage locally; a failure returns no candidate and changes no
    input state. No selected pair can see another pair's candidate edits.
    """
    original, candidate = _prepare(parent, expected_parent_id, adapter_limits)
    if not isinstance(programs, dict) or len(programs) > adapter_limits.max_pairs:
        raise ValueError('A projection-to-source mapping is required')
    totals = dict(operations=0, statements=0, loop_iterations=0); edits = []
    for projection, source in sorted(programs.items()):
        if not isinstance(source, str) or len(source.encode('utf8')) > adapter_limits.max_program_bytes:
            raise ValueError('WSPL program size limit')
        tree = parse(source)
        if (not tree['edit'] or tree['edit']['mode'] != 'fixed'
                or tree['edit']['numeric'] != 'exact_to_storage'
                or tree['circuits'] or tree['regions'] or tree['descriptions']):
            raise ValueError('Fixed exact-to-storage tensor construction required')
        pair = _pair(original, projection, adapter_limits)
        staged = plan(source, parent=pair, limits=editor_limits)
        result = apply(staged, parent=pair, expected_parent_id=pair.sha256)
        detail = staged.to_dict()
        if result.graph != pair.graph:
            raise ValueError('LoRA factor graph cannot change')
        for name in totals:
            totals[name] += detail['resources'][name]
            if totals[name] > getattr(adapter_limits, 'max_total_'+name):
                raise ValueError('Aggregate construction resource limit exceeded')
        values = result.arrays()
        for factor in ('A', 'B'):
            candidate[f'{projection}.lora_{factor}'] = values[factor].copy()
        edits.append(dict(projection=projection, source=source, plan_id=staged.sha256,
                          pair_parent_id=pair.sha256, pair_candidate_id=result.sha256,
                          resources=detail['resources']))
    state, receipt = _finish(original, candidate, edits, 'wspl', adapter_limits)
    receipt['resources'] = totals
    return state, receipt


def construct_native(parent, replacements, *, expected_parent_id, limits=AdapterLimits()):
    """Native comparator: complete selected A/B replacements, same final checks."""
    original, candidate = _prepare(parent, expected_parent_id, limits)
    if not isinstance(replacements, dict) or len(replacements) > limits.max_pairs:
        raise ValueError('A projection-to-factor mapping is required')
    edits = []
    for projection, factors in sorted(replacements.items()):
        _pair(original, projection, limits)
        if not isinstance(factors, dict) or set(factors) != {'A', 'B'}:
            raise ValueError('Both replacement factors required')
        for factor, value in factors.items():
            key = f'{projection}.lora_{factor}'
            if not isinstance(value, np.ndarray) or value.shape != original[key].shape:
                raise ValueError('Replacement factor shape differs')
            candidate[key] = value.copy()
        edits.append(dict(projection=projection))
    return _finish(original, candidate, edits, 'native', limits)


def model_state(model, limits=AdapterLimits()):
    """Trusted host adapter extraction; all trainable parameters must be factors."""
    return snapshot({name: value.detach().cpu().numpy().copy()
                     for name, value in model.named_parameters() if value.requires_grad}, limits)


def install_state(model, state, *, expected_current_id, limits=AdapterLimits()):
    """Validated adapter-only replacement. Base weights and scale are never written."""
    import torch
    before = model_state(model, limits)
    if identity(before, limits) != expected_current_id:
        raise ValueError('Live model adapter identity differs')
    candidate, _ = _finish(before, state, [], 'install', limits)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    # All validation precedes writes. Roll back unexpected device-copy failures.
    try:
        with torch.no_grad():
            for name, array in candidate.items():
                parameters[name].copy_(torch.from_numpy(array).to(parameters[name].device))
    except BaseException:
        with torch.no_grad():
            for name, array in before.items():
                parameters[name].copy_(torch.from_numpy(array).to(parameters[name].device))
        raise
    return identity(candidate, limits)
