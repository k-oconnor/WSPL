"""Matched data-only construction workers; no candidate code is executed."""
import hashlib
import json
import re
import numpy as np
from .robot_array_language import Calculator
from .wspl_lora import (describe, identity, construct_native, construct_wspl,
                        source_for_pair, snapshot)
from .wsp_artifact import Bits, digest
from .wsp_values import pack


def aliases(parent):
    projections = sorted({k.rsplit('.', 1)[0] for k in parent})
    return {p: {f: f'P{i:03d}{f}' for f in ('A', 'B')} for i, p in enumerate(projections)}


def numerical_inputs(parent, features):
    inputs = dict(features)
    for p, names in aliases(parent).items():
        for f, name in names.items():
            inputs[name] = parent[p+'.lora_'+f]
    return inputs


def summaries(arrays):
    return {k: dict(shape=list(a.shape), dtype=str(a.dtype),
                     sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest(),
                     norm=float(np.linalg.norm(a.astype(np.float64))),
                     preview=a.reshape(-1)[:12].tolist()) for k, a in arrays.items()}


def construct(parent, features, request, arm):
    if arm not in ('native', 'wspl') or set(request) != {'tool', 'compute', 'patches'} \
            or request['tool'] != 'propose':
        raise ValueError('Expected tool=propose, compute, patches')
    computation = request['compute']
    if not isinstance(computation, dict) or set(computation) != {'source', 'outputs'}:
        raise ValueError('compute requires source and outputs')
    exported, numeric_receipt = Calculator(numerical_inputs(parent, features)).run(
        computation['source'], computation['outputs'])
    if any(k in ('A', 'B') or k.startswith('P') for k in exported):
        raise ValueError('Construction exports cannot shadow factor aliases')
    if sum(a.size for a in exported.values()) > 2_000_000 or any(a.size > 1_000_000 for a in exported.values()):
        raise ValueError('Construction binding storage limit')
    patches = request['patches']
    if not isinstance(patches, dict) or not patches or len(patches) > 196:
        raise ValueError('One to 196 projection bodies required')
    if sum(len(x.encode()) for x in patches.values() if isinstance(x, str)) > 2_000_000:
        raise ValueError('Total construction source limit')
    parent_id = identity(parent)
    replacements, programs = {}, {}
    if arm == 'native':
        for p, body in patches.items():
            if p+'.lora_A' not in parent or not isinstance(body, str):
                raise ValueError('Unknown projection or invalid source')
            inputs = dict(exported, A=parent[p+'.lora_A'], B=parent[p+'.lora_B'])
            factors, _ = Calculator(inputs).run(body, ['A', 'B'])
            replacements[p] = factors
        candidate, receipt = construct_native(parent, replacements, expected_parent_id=parent_id)
    else:
        bindings = {k: np.array([Bits(int(v)) for v in a.view(np.uint32).flat], dtype=object).reshape(a.shape)
                    for k, a in exported.items()}
        declarations = '\n'.join(f'const {k}=tensor("{k}",sha256="{digest(pack(v))}");'
                                 for k, v in bindings.items())
        catalog = describe(parent)
        for p, body in patches.items():
            if p not in catalog['projections'] or not isinstance(body, str):
                raise ValueError('Unknown projection or invalid source')
            source = source_for_pair(catalog['projections'][p]['pair_id'], body)
            programs[p] = source.replace('edit adapter', declarations+'\nedit adapter', 1)
        candidate, receipt = construct_wspl(parent, programs, expected_parent_id=parent_id, bindings=bindings)
    effective = {}
    # Small Gram products give the exact algebraic Frobenius norm without
    # materializing each dense delta: ||B1 A1 - B0 A0||_F^2.
    for p in patches:
        a0, b0 = [parent[p+'.lora_'+f].astype(np.float64) for f in ('A', 'B')]
        a1, b1 = [candidate[p+'.lora_'+f].astype(np.float64) for f in ('A', 'B')]
        b, a = np.concatenate([b1, -b0], axis=1), np.concatenate([a1, a0], axis=0)
        squared = float(np.sum((b.T@b)*(a@a.T)))
        effective[p] = math_sqrt_nonnegative(squared)
    receipt.update(compute_source=computation['source'], compute_resources=numeric_receipt,
                   exported_arrays=summaries(exported), effective_unscaled_delta_norm=effective,
                   shared_numerical_construction=True,
                   attribution='Audit compute and patch source; WSPL transport alone is not evidence of WSPL-specific synthesis.')
    return candidate, receipt, exported


def math_sqrt_nonnegative(value):
    # Cancellation for functionally equivalent factor changes can leave tiny
    # negative roundoff; this diagnostic is not an exact symbolic certificate.
    if not np.isfinite(value):
        raise ValueError('Nonfinite effective-update diagnostic')
    return float(np.sqrt(max(0., value)))
