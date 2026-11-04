"""Reworked WSPL versus native edits on the original fixed tanh Breakout network.

Construction and frozen-artifact checks only; no model calls or game execution.
"""
import re

import numpy as np

from .csail_breakout import BreakoutPolicy
from .native_weight_edits import NativeEditedPolicy, NAMES, statements
from .wsp_artifact import TensorArtifact, digest, inspect
from .wsp_editor import plan, apply
from .wsp_evidence import equations
from .wsp_syntax import parse
from .wsp_values import Limits

ARMS = ('native', 'wspl')
GRAPH = dict(kind='dense', widths=[600, 32, 32, 6],
             activations=['tanh', 'tanh', 'linear'], layout='in_out')
LIMITS = Limits(max_elements=20486, max_operations=100_000_000,
                max_statements=4096, max_loop_iterations=4096)
HEADER = ('wspl 0.2;\ntarget base=dense(widths=[600,32,32,6], '
          'activations=[tanh,tanh,linear],layout=in_out,storage=f32,init=zeros);\n')


def zero_parent():
    return TensorArtifact.dense([600,32,32,6], ['tanh','tanh','linear'])


def native_source(body):
    rows = list(statements(body))
    return HEADER + 'edit base as candidate mode=fixed numeric=native_numpy_v1 {\n' + '\n'.join(r+';' for r in rows) + '\n}\n'


def fenced(answer, tag):
    if not isinstance(answer, str) or len(answer.encode('utf-8')) > 1_000_000:
        raise ValueError('reply must be text of at most 1000000 UTF-8 bytes')
    blocks = re.findall(r'```([^\r\n`]*)\r?\n(.*?)```', answer, re.S)
    if len(blocks) != 1 or blocks[0][0].strip() != tag:
        raise ValueError('exactly one fenced '+tag+' candidate required')
    return blocks[0][1]


def prompt(native_text, guide, arm):
    if arm not in ARMS:
        raise ValueError('unknown arm')
    game = native_text[native_text.index('THE GAME:'):native_text.index('THE NETWORK (')]
    return ('Author from scratch a neural Breakout policy through the selected interface. '
            'One fresh response; no tools, code execution, training, evaluation feedback or retries. '
            'Briefly explain your strategy and hidden-unit roles, then supply exactly one candidate block.\n\n'
            + game + '\nCOMMON NETWORK AND INTERFACE GUIDE\n' + guide
            + '\n\nSELECTED ARM: ' + arm + '\n')


class ReworkedBreakoutPolicy(BreakoutPolicy):
    def __init__(self):
        self.parent = zero_parent()
        self._constructed = False

    def construct(self, arm, answer):
        if self._constructed or arm not in ARMS:
            raise ValueError('one proposal per fresh policy; known arm required')
        self._constructed = True
        native = None
        if arm == 'native':
            body = fenced(answer, 'edits')
            native = NativeEditedPolicy()
            parsed = native.apply_answer(answer)
            if parsed['skipped']:
                raise ValueError('native proposal contains invalid statements: '+str(parsed['errors']))
            source = native_source(body)
        else:
            source = fenced(answer, 'wspl')
            tree = parse(source)
            if tree['edit'] is None or tree['edit']['mode'] != 'fixed' or tree['regions'] or tree['circuits'] or tree['descriptions']:
                raise ValueError('fixed tensor program required; tanh placement and graph augmentation are outside this screen')
            parsed = None
        edit_plan = plan(source, parent=self.parent, limits=LIMITS)
        artifact = apply(edit_plan, parent=self.parent, expected_parent_id=self.parent.sha256)
        if artifact.to_dict()['graph'] != GRAPH:
            raise ValueError('the original fixed tanh architecture is required')
        arrays = artifact.arrays()
        if any(not np.isfinite(a).all() or (np.abs(a)>1e6).any() for a in arrays.values()):
            raise ValueError('all stored weights must be finite and within +/-1000000')
        self.weights = {NAMES[k]:a.copy() for k,a in arrays.items()}
        for a in self.weights.values():
            a.flags.writeable = False
        if native is not None and self.tensor_identity() != native.tensor_identity():
            raise ValueError('native parser/editor compatibility failed')
        self.mode = arm
        self.artifact_sha256 = artifact.sha256
        receipt = dict(arm=arm, candidate_id=artifact.sha256, parent_id=self.parent.sha256,
                       tensor_sha256=self.tensor_identity(), parameters=sum(a.size for a in arrays.values()),
                       nonzero=sum(int(np.count_nonzero(a)) for a in arrays.values()),
                       native_parse=parsed, numeric_profile=edit_plan.to_dict()['numeric_profile'],
                       plan_id=edit_plan.sha256, resources=edit_plan.to_dict()['resources'],
                       inspection=inspect(artifact), formal_status='UNSUPPORTED_TANH',
                       semantic_roles='author hypotheses; not verified')
        return artifact, edit_plan, receipt


def frozen_checks(policy, artifact):
    """Deployment-adapter parity on a disclosed synthetic panel, not game success."""
    rng = np.random.default_rng(913170)
    rows = np.concatenate([np.zeros((1,600),np.float32), np.ones((1,600),np.float32),
                           rng.integers(0,2,size=(30,600)).astype(np.float32)])
    maximum = 0.
    for row in rows:
        a = policy.logits(row)
        b = np.asarray(artifact.trace(row,semantics='numpy_f32')['y'],np.float32)
        if not np.array_equal(a.view(np.uint32),b.view(np.uint32)):
            raise ValueError('artifact/deployment inference differs on frozen panel')
        maximum = max(maximum, float(np.max(np.abs(a-b))))
    return dict(status='EMPIRICAL_PARITY', candidate_id=artifact.sha256,
                fixture_id=digest(rows.tolist()), samples=len(rows), max_absolute_error=maximum,
                universal_ieee_claim=False, formal_status='UNSUPPORTED_TANH')


def stored_equations(artifact):
    return dict(candidate_id=artifact.sha256,
                layers=[equations(artifact,layer=i,limit=32) for i in (1,2,3)],
                semantic_interpretation='not_established', formal_status='UNSUPPORTED_TANH')
