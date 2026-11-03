"""Physical task specifications, asset inspection and symbolic trace fixtures.

No simulator construction, policy, training, planner or physical observer is
implemented here. A matching labelled trace is not evidence labels are grounded.
"""
import copy
import hashlib
import math
import re
from pathlib import Path, PurePosixPath


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', value):
        raise ValueError('invalid identifier')
    return value


def names(values):
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
        raise ValueError('expected unique name list')
    return [identifier(v) for v in values]


class RobotTaskSuite:
    def __init__(self, document):
        d = copy.deepcopy(document)
        if set(d) != {'format', 'clock_hz', 'status', 'skills', 'capabilities', 'tasks'}:
            raise ValueError('unexpected suite fields')
        if d['format'] != 'skillforge.robot_tasks.v1' or d['clock_hz'] != 20 or d['status'] != 'inspected_design_only':
            raise ValueError('unsupported suite contract')
        self._d = d
        if not isinstance(d['skills'], dict) or not d['skills'] or not isinstance(d['capabilities'], dict):
            raise ValueError('invalid registries')
        for registry in ('skills', 'capabilities'):
            for key, item in d[registry].items():
                identifier(key)
                if set(item) != {'status', 'meaning'} or item['status'] not in ('available', 'planned') or not item['meaning']:
                    raise ValueError('invalid capability/skill status')
        if not isinstance(d['tasks'], list) or not d['tasks']:
            raise ValueError('empty task suite')
        self._tasks = {}
        for task in d['tasks']:
            self._validate(task)
            if task['id'] in self._tasks:
                raise ValueError('duplicate task ID')
            self._tasks[task['id']] = task

    def _validate(self, t):
        if set(t) != {'id', 'tier', 'family', 'instruction', 'axes', 'scene', 'bindings',
                      'requirements', 'milestones', 'precedence', 'terminal', 'invariants',
                      'dwell', 'max_motor_ticks', 'expected_grasp_occurrences', 'perturbation'}:
            raise ValueError('unexpected task fields')
        identifier(t['id']); identifier(t['family']); names(t['axes'])
        if type(t['tier']) is not int or t['tier'] not in (1, 2, 3, 4) or not t['instruction']:
            raise ValueError('invalid task tier/instruction')
        if type(t['max_motor_ticks']) is not int or t['max_motor_ticks'] < 1:
            raise ValueError('invalid motor budget')
        scene = t['scene']
        if set(scene) != {'asset', 'mode', 'changes', 'planned_entities'}:
            raise ValueError('invalid scene contract')
        asset = PurePosixPath(scene['asset'])
        if (asset.is_absolute() or '..' in asset.parts or '\\' in str(asset) or
            not str(asset).startswith('libero/libero/bddl_files/') or asset.suffix != '.bddl'):
            raise ValueError('unsafe scene asset path')
        if scene['mode'] not in ('existing_scene', 'goal_overlay', 'custom_scene_planned'):
            raise ValueError('unknown scene mode')
        if (not isinstance(scene['changes'], list) or any(not isinstance(c, str) or not c for c in scene['changes']) or
            (scene['mode'] != 'existing_scene' and not scene['changes'])):
            raise ValueError('scene changes must be explicit')
        if scene['mode'] == 'existing_scene' and (scene['changes'] or scene['planned_entities']):
            raise ValueError('existing scene cannot claim unbuilt changes')
        if not isinstance(scene['planned_entities'], dict):
            raise ValueError('invalid planned entities')
        for key, meaning in scene['planned_entities'].items():
            identifier(key)
            if not meaning or scene['mode'] != 'custom_scene_planned':
                raise ValueError('new geometry requires a custom scene')
        if not isinstance(t['bindings'], dict) or not t['bindings']:
            raise ValueError('empty bindings')
        for k, v in t['bindings'].items(): identifier(k); identifier(v)
        req = t['requirements']
        if set(req) != {'skills', 'capabilities'}:
            raise ValueError('invalid requirements')
        for key in req:
            if set(names(req[key])) - set(self._d[key]):
                raise ValueError('unknown required skill/capability')
        events = []
        roles = dict(pickup={'object'}, place={'object', 'target'}, open={'fixture'}, close={'fixture'},
                     stack={'object', 'target'}, pass_gate={'object', 'gate'}, fault={'object'})
        if not isinstance(t['milestones'], list): raise ValueError('expected milestone list')
        for event in t['milestones']:
            if set(event) != {'id', 'kind', 'bindings'} or event['kind'] not in (
                    'pickup', 'place', 'open', 'close', 'stack', 'pass_gate', 'fault'):
                raise ValueError('invalid milestone')
            events.append(identifier(event['id']))
            if (not isinstance(event['bindings'], dict) or set(event['bindings']) != roles[event['kind']] or
                set(event['bindings'].values()) - set(t['bindings'])):
                raise ValueError('milestone references unknown role')
        if not events or len(set(events)) != len(events):
            raise ValueError('milestone IDs must denote distinct observed occurrences')
        if type(t['expected_grasp_occurrences']) is not int or t['expected_grasp_occurrences'] != sum(e['kind'] == 'pickup' for e in t['milestones']):
            raise ValueError('grasp count must match declared pickup occurrences')
        terminal = names(t['terminal']); invariants = names(t['invariants'])
        if not terminal or not invariants or set(events) & set(terminal + invariants):
            raise ValueError('separate event and state propositions required')
        edges = t['precedence']
        if not isinstance(edges, list) or any(not isinstance(e, list) or len(e) != 2 for e in edges):
            raise ValueError('invalid precedence edges')
        if len(set(tuple(e) for e in edges)) != len(edges):
            raise ValueError('duplicate precedence edge')
        for edge in edges:
            if set(edge) - set(events) or edge[0] == edge[1]:
                raise ValueError('unknown/self precedence edge')
        if not isinstance(t['dwell'], list): raise ValueError('expected dwell list')
        for dwell in t['dwell']:
            if set(dwell) != {'after', 'before', 'state', 'ticks'} or dwell['after'] not in events or dwell['before'] not in events:
                raise ValueError('invalid dwell anchors')
            identifier(dwell['state'])
            if (dwell['state'] in events or dwell['after'] == dwell['before'] or type(dwell['ticks']) is not int or
                not 1 <= dwell['ticks'] < t['max_motor_ticks']):
                raise ValueError('invalid dwell state/budget')
        done = set()
        order = edges + [[d['after'], d['before']] for d in t['dwell']]
        while len(done) < len(events):
            ready = {e for e in events if e not in done and all(a in done for a, b in order if b == e)}
            if not ready: raise ValueError('cyclic precedence/dwell')
            done |= ready
        if 'time_s' in set(events + terminal + invariants + [d['state'] for d in t['dwell']]):
            raise ValueError('time_s is reserved trace metadata')
        perturb = t['perturbation']
        if perturb is not None:
            if set(perturb) != {'after', 'kind', 'count', 'scope'} or perturb['after'] not in events or perturb['count'] != 1:
                raise ValueError('invalid scheduled perturbation')
            if not perturb['kind'] or not perturb['scope'] or scene['mode'] != 'custom_scene_planned':
                raise ValueError('perturbation requires explicit custom integration')

    @property
    def task_ids(self):
        return tuple(self._tasks)

    def task(self, task_id):
        return copy.deepcopy(self._tasks[task_id])

    def readiness(self, task_id):
        t = self._tasks[task_id]
        missing = [f'{kind}:{key}' for kind, keys in t['requirements'].items()
                   for key in keys if self._d[kind][key]['status'] != 'available']
        if t['scene']['mode'] == 'custom_scene_planned': missing.append('scene:custom_construction')
        # These specs have no physical executor binding, even with available parts.
        return dict(physical_runnable=False, missing=missing, reason='task executor/observers not integrated')

    def check_trace(self, task_id, trace):
        """Check externally labelled *closed* motor traces; not physical grounding.

        Event flags must pulse once per explicitly named occurrence. Strict edges
        forbid even an early wrong-order occurrence followed by a later retry.
        Final states must hold at STOP; safety predicates hold at every sample.
        Dwell counts consecutive observed ticks strictly between its anchors.
        """
        t = self._tasks[task_id]
        events = [e['id'] for e in t['milestones']]
        props = set(events + t['terminal'] + t['invariants'] + [d['state'] for d in t['dwell']])
        if not isinstance(trace, list) or not trace:
            raise ValueError('nonempty closed trace required')
        if any(not isinstance(row, dict) or any(p not in row or type(row[p]) is not bool for p in props) for row in trace):
            raise ValueError('all required observations must be explicit Booleans')
        for i, row in enumerate(trace):
            timestamp = row.get('time_s')
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError('explicit finite time_s required')
            if i and not math.isclose(timestamp-trace[i-1]['time_s'], 1/self._d['clock_hz'], rel_tol=0, abs_tol=1e-6):
                raise ValueError('consecutive 20 Hz observations required')
        violations = []
        if len(trace)-1 > t['max_motor_ticks']: violations.append('motor_budget')
        seen = {}
        for event in events:
            positions = [i for i, row in enumerate(trace) if row[event]]
            if len(positions) != 1: violations.append('event_occurrence:'+event)
            else: seen[event] = positions[0]
        for a, b in t['precedence']:
            if a in seen and b in seen and seen[a] >= seen[b]: violations.append('precedence:'+a+'<'+b)
        for p in t['terminal']:
            if not trace[-1][p]: violations.append('terminal:'+p)
        for p in t['invariants']:
            if not all(row[p] for row in trace): violations.append('invariant:'+p)
        for d in t['dwell']:
            if d['after'] in seen and d['before'] in seen:
                a, b = seen[d['after']], seen[d['before']]
                if b-a-1 < d['ticks'] or not all(trace[i][d['state']] for i in range(a+1, min(b, a+d['ticks']+1))):
                    violations.append('dwell:'+d['state'])
        return dict(accepted=not violations, violations=violations, scope='label semantics only')


def read_bddl(text):
    """Bounded S-expression inspection; not a replacement for LIBERO parsing."""
    if len(text) > 1_000_000: raise ValueError('asset too large')
    tokens = re.findall(r'\(|\)|[^\s()]+', re.sub(r';[^\n]*', '', text))
    if len(tokens) > 100_000: raise ValueError('too many asset tokens')
    stack = [[]]
    for token in tokens:
        if token == '(':
            node = []; stack[-1].append(node); stack.append(node)
            if len(stack) > 64: raise ValueError('asset nesting limit')
        elif token == ')':
            if len(stack) == 1: raise ValueError('unbalanced asset')
            stack.pop()
        else: stack[-1].append(token)
    if len(stack) != 1 or len(stack[0]) != 1 or not stack[0][0] or stack[0][0][0] != 'define':
        raise ValueError('invalid BDDL root')
    root = stack[0][0]
    def section(name):
        matches = [x[1:] for x in root if isinstance(x, list) and x and x[0] == name]
        if len(matches) > 1: raise ValueError('duplicate BDDL section')
        return matches[0] if matches else []
    entities = set()
    for name in (':fixtures', ':objects'):
        items = section(name); pending = []; i = 0
        while i < len(items):
            if not isinstance(items[i], str): raise ValueError('invalid typed object declaration')
            if items[i] == '-':
                if not pending or i+1 >= len(items) or not isinstance(items[i+1], str): raise ValueError('invalid typed objects')
                entities.update(pending); pending = []; i += 2
            else: pending.append(items[i]); i += 1
        if pending: raise ValueError('untyped objects')
    for region in section(':regions'):
        if not isinstance(region, list) or not region: raise ValueError('invalid region')
        targets = [x for x in region[1:] if isinstance(x, list) and len(x) == 2 and x[0] == ':target']
        if len(targets) != 1: raise ValueError('invalid region target')
        entities.add(targets[0][1]+'_'+region[0])
    goal = section(':goal')
    if len(goal) != 1: raise ValueError('missing/ambiguous native goal')
    return dict(entities=sorted(entities), native_goal=goal[0])


def audit_assets(suite, checkout):
    checkout = Path(checkout).resolve(strict=True)
    results = []
    for task_id in suite.task_ids:
        t = suite.task(task_id); relative = t['scene']['asset']
        path = (checkout/relative).resolve(strict=True)
        if not path.is_relative_to(checkout): raise ValueError('asset escapes checkout')
        raw = path.read_bytes(); asset = read_bddl(raw.decode('utf-8'))
        planned = set(t['scene']['planned_entities'])
        missing = set(t['bindings'].values()) - set(asset['entities']) - planned
        if missing: raise ValueError(f'{task_id}: unknown native bindings {sorted(missing)}')
        results.append(dict(id=task_id, asset=relative, sha256=hashlib.sha256(raw).hexdigest(),
                            native_goal=asset['native_goal'], scene_mode=t['scene']['mode'],
                            planned_entities=sorted(planned), readiness=suite.readiness(task_id)))
    return dict(scope='read-only asset/specification audit; no simulation', tasks=results)
