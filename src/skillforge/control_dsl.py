"""Typed, bounded skill sequencing over frozen neural operators.

Engineering workcell only: discrete kinematics, no contact physics or perception.
The interpreter does not learn routes. Neural functions implement motor commands
and predicate proposals; independent observations validate completion.
"""
from dataclasses import dataclass, field
import math
import re

from .controller_author import predict


def point(value):
    if len(value) != 2 or not all(math.isfinite(v) for v in value):
        raise ValueError('Expected a finite planar point')
    return tuple(float(v) for v in value)


@dataclass(frozen=True)
class Call:
    operator: str
    argument: str


def compile_program(source, objects, points, limit=128):
    """Small line language; transfer expands without duplicating block weights."""
    if set(objects) & set(points):
        raise ValueError('Object and point namespaces overlap')
    calls = []
    for line_number, raw in enumerate(source.splitlines(), 1):
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'(move_to|grasp|release|transfer)\s+([A-Za-z_]\w*)(?:\s+to\s+([A-Za-z_]\w*))?', line)
        if not match:
            raise ValueError(f'Invalid statement at line {line_number}')
        op, arg, destination = match.groups()
        if op == 'transfer':
            if arg not in objects or destination not in points:
                raise ValueError('transfer requires Object to Point')
            calls.extend([Call('move_to', arg), Call('grasp', arg),
                          Call('move_to', destination), Call('release', arg)])
        else:
            if destination is not None:
                raise ValueError('Only transfer accepts a destination')
            allowed = set(objects) | set(points) if op == 'move_to' else set(objects)
            if arg not in allowed:
                raise ValueError(f'Invalid argument type for {op}: {arg}')
            calls.append(Call(op, arg))
        if len(calls) > limit:
            raise ValueError('Expanded instruction budget exceeded')
    if not calls:
        raise ValueError('Empty program')
    return tuple(calls)


@dataclass
class Workcell:
    position: tuple
    objects: dict
    held: str | None = None
    radius: float = .04
    bound: float = 1.
    jammed: frozenset = frozenset()

    def __post_init__(self):
        self.position = point(self.position)
        self.objects = {name: point(p) for name, p in self.objects.items()}
        if not 0 < self.radius < self.bound or not math.isfinite(self.bound):
            raise ValueError('Invalid physical geometry')
        if self.held is not None and self.held not in self.objects:
            raise ValueError('Unknown held object')

    def near(self, target):
        return max(abs(a-b) for a, b in zip(self.position, target)) <= self.radius + 1e-9

    def safe(self):
        return max(map(abs, self.position)) <= self.bound

    def apply(self, command):
        """Contact follows geometry, never the program's requested object name."""
        if set(command) != {'vx', 'vy', 'grip'} or not all(math.isfinite(v) for v in command.values()):
            raise ValueError('Invalid actuator command')
        if max(abs(command['vx']), abs(command['vy'])) > .2 or abs(command['grip']) > 1:
            raise ValueError('Actuator range exceeded')
        self.position = tuple(a + command[k] for a, k in zip(self.position, ('vx', 'vy')))
        if command['grip'] > .5 and self.held is None:
            contacts = [name for name, p in self.objects.items() if self.near(p) and name not in self.jammed]
            if len(contacts) == 1:
                self.held = contacts[0]
        if self.held is not None:
            self.objects[self.held] = self.position
        if command['grip'] < -.5:
            self.held = None

    def observe(self):
        return dict(position=list(self.position), objects={k: list(v) for k, v in self.objects.items()},
                    held=self.held, safe=self.safe())


@dataclass
class Executor:
    program: tuple
    points: dict
    blocks: dict
    skill_budget: int = 160
    total_budget: int = 2048
    pc: int = 0
    skill_ticks: int = 0
    ticks: int = 0
    status: str = 'running'
    reason: str | None = None
    trace: list = field(default_factory=list)

    def __post_init__(self):
        self.points = {name: point(p) for name, p in self.points.items()}
        if self.skill_budget < 1 or self.total_budget < 1:
            raise ValueError('Budgets must be positive')
        if set(self.blocks) != {'move', 'near', 'holding', 'close', 'open', 'safe'}:
            raise ValueError('Incomplete block library')

    def tick(self, world):
        if self.status != 'running':
            return self.status
        before = world.observe()
        event = dict(pc=self.pc, tick=self.ticks, before=before, calls=[])

        def fail(reason):
            self.status, self.reason = 'failure', reason

        def neural(kind, values):
            event['calls'].append(dict(block=kind, inputs=values))
            result = predict(self.blocks[kind], values)
            if not all(math.isfinite(v) for v in result.values()):
                raise ValueError('Nonfinite block output')
            event['calls'][-1]['outputs'] = result
            return result

        try:
            if self.ticks >= self.total_budget or self.skill_ticks >= self.skill_budget:
                fail('budget_exhausted')
            elif not world.safe():
                fail('physical_safety_violation')
            elif neural('safe', dict(x=world.position[0], y=world.position[1]))['score'] < 0:
                fail('safety_predicate_stop')
            else:
                call = self.program[self.pc]
                event['instruction'] = dict(operator=call.operator, argument=call.argument)
                complete = False
                command = None
                if call.operator == 'move_to':
                    target = world.objects[call.argument] if call.argument in world.objects else self.points[call.argument]
                    errors = dict(x=target[0]-world.position[0], y=target[1]-world.position[1])
                    if neural('near', errors)['score'] >= 0:
                        if world.near(target):
                            complete = True
                        else:
                            fail('false_completion_predicate')
                    else:
                        command = neural('move', errors)
                elif call.operator == 'grasp':
                    if world.held is not None or not world.near(world.objects[call.argument]):
                        fail('grasp_precondition')
                    else:
                        command = neural('close', dict(x=0., y=0.))
                elif call.operator == 'release':
                    if world.held != call.argument:
                        fail('release_precondition')
                    else:
                        command = neural('open', dict(x=0., y=0.))
                else:
                    raise ValueError('Unknown operator')
                if command is not None:
                    event['action'] = command
                    world.apply(command)
                    if not world.safe():
                        fail('physical_safety_violation')
                    if call.operator in ('grasp', 'release') and self.status == 'running':
                        claimed = neural('holding', dict(h=int(world.held is not None)))['score'] >= 0
                        actual = world.held == call.argument if call.operator == 'grasp' else world.held is None
                        proposed = claimed if call.operator == 'grasp' else not claimed
                        if proposed and not actual:
                            fail('false_completion_predicate')
                        elif proposed and actual:
                            complete = True
                if complete and self.status == 'running':
                    event['skill_status'] = 'success'
                    self.pc += 1
                    self.skill_ticks = -1
                    if self.pc == len(self.program):
                        self.status = 'success'
        except (ValueError, KeyError, IndexError) as error:
            fail('contract_error: ' + str(error))
        self.ticks += 1
        self.skill_ticks += 1
        event.update(after=world.observe(), status=self.status, reason=self.reason)
        event.setdefault('skill_status', 'failure' if self.status == 'failure' else 'running')
        self.trace.append(event)
        return self.status

    def run(self, world):
        while self.status == 'running':
            self.tick(world)
        return dict(status=self.status, reason=self.reason, ticks=self.ticks,
                    completed_instructions=self.pc, trace=self.trace)
