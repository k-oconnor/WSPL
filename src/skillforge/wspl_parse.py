"""Text front end for the WSPL Core PWA module form (proposal v0.1, milestone M0).

Parses the proposal's `module Name { ... }` surface into the existing
wspl.scalar_ir.v0 JSON and wspl.io_contract_suite.v0 contracts. Only that module
form and Core PWA expressions are accepted; unsupported profiles are rejected
with a line:column error, never approximated. The produced IR is validated by
ScalarProgram. Parsing establishes no contract, equivalence or execution claim.
"""
import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sys
from .wspl_core import ScalarProgram
from .wspl_artifact import ReLUArtifact, canonical_bytes

MAX_SOURCE_BYTES = 1_000_000
MAX_NESTING = 64
UNIT_RULE = ('labels starting with normalized_ denote explicitly normalized dimensionless '
             'quantities (IR unit "1"); other labels are physical units kept verbatim')
UNSUPPORTED = {'requires', 'invariant', 'empirical', 'state', 'function', 'import', 'tensor'}
RESERVED = {'module', 'input', 'output', 'let', 'ensures', 'preserves', 'forall', 'inputs',
            'Real', 'Truth', 'true', 'false', 'relu', 'clip', 'margin', 'min', 'max',
            'not', 'and', 'or', 'implies'} | UNSUPPORTED
FUNCTIONS = {'relu': 1, 'not': 1, 'clip': 3, 'margin': 3, 'min': 2, 'max': 2,
             'and': 2, 'or': 2, 'implies': 2}
COMPARISONS = {'<=': 'le', '<': 'lt', '>=': 'ge', '>': 'gt', '==': 'eq'}
_TOKEN = re.compile(r'(?P<space>[ \t\r]+)|(?P<newline>\n)|(?P<comment>//[^\n]*)'
                    r'|(?P<number>\d+(?:\.\d+)?|\.\d+)|(?P<name>[A-Za-z_][A-Za-z0-9_]*)'
                    r'|(?P<symbol>==|<=|>=|[{}()\[\],;:=<>+\-*/])')
_PORT_NAME = re.compile(r'[a-z][a-z0-9_]*')


def ir_unit(label):
    """Map a declared unit label to an IR unit (reading of the proposal; issue #3)."""
    return '1' if label.startswith('normalized_') else label


class ParseError(ValueError):
    def __init__(self, message, position=None):
        self.position = position
        super().__init__(f'{position[0]}:{position[1]}: {message}' if position else message)


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    position: tuple


def _q(value):
    return str(Fraction(value))


def tokenize(text):
    tokens, line, start, index = [], 1, 0, 0
    while index < len(text):
        match = _TOKEN.match(text, index)
        if match is None:
            raise ParseError(f'unexpected character {text[index]!r}', (line, index - start + 1))
        kind, value = match.lastgroup, match.group()
        if kind == 'newline':
            line, start = line + 1, match.end()
        elif kind not in ('space', 'comment'):
            tokens.append(Token(kind, value, (line, index - start + 1)))
        index = match.end()
    tokens.append(Token('eof', '', (line, index - start + 1)))
    return tokens


class _Linear:
    """Exact linear combination of IR atoms; constants have no terms."""

    def __init__(self, terms=None, bias=Fraction(0)):
        self.terms = dict(terms or {})  # key -> (coefficient, ir, unit)
        self.bias = Fraction(bias)

    @classmethod
    def atom(cls, ir, unit):
        return cls({json.dumps(ir, sort_keys=True): (Fraction(1), ir, unit)})

    def constant(self):
        return all(c == 0 for c, _, _ in self.terms.values())

    def unit(self):
        return next((u for c, _, u in self.terms.values() if c != 0), None)

    def scale(self, factor):
        return _Linear({k: (c * factor, ir, u) for k, (c, ir, u) in self.terms.items()}, self.bias * factor)

    def plus(self, other, position):
        a, b = self.unit(), other.unit()
        if a is not None and b is not None and a != b:
            raise ParseError(f"cannot add unit {a!r} to unit {b!r}; normalize explicitly", position)
        terms = dict(self.terms)
        for key, (c, ir, u) in other.terms.items():
            terms[key] = (terms[key][0] + c, ir, u) if key in terms else (c, ir, u)
        return _Linear(terms, self.bias + other.bias)

    def to_ir(self, context_unit):
        terms = [(c, ir, u) for c, ir, u in self.terms.values() if c != 0]
        if not terms:
            unit = context_unit or '1'
            return {'op': 'const', 'value': _q(self.bias), 'unit': unit}, unit
        if len(terms) == 1 and terms[0][0] == 1 and self.bias == 0:
            return terms[0][1], terms[0][2]
        return ({'op': 'affine', 'terms': [{'coefficient': _q(c), 'value': ir} for c, ir, _ in terms],
                 'bias': _q(self.bias), 'unit': terms[0][2]}, terms[0][2])


@dataclass
class ParsedModule:
    name: str
    program: dict
    contracts: list
    report: dict

    def contract_suite(self, artifact):
        """Bind the parsed contracts to an artifact compiled from this module's IR."""
        if not isinstance(artifact, ReLUArtifact):
            artifact = ReLUArtifact(artifact)
        if artifact.to_dict()['source_sha256'] != self.report['ir_sha256']:
            raise ValueError('artifact was not compiled from this module')
        if not self.contracts:
            raise ValueError('module declares no contracts')
        return {'format': 'wspl.io_contract_suite.v0', 'artifact_sha256': artifact.sha256,
                'contracts': [{k: c[k] for k in ('name', 'assumptions', 'ensures')} for c in self.contracts]}


class _Parser:
    def __init__(self, text, *, max_nodes=1024):
        self.max_nodes=max_nodes
        if not isinstance(text, str):
            raise ParseError('WSPL source text required')
        if len(text.encode('utf-8')) > MAX_SOURCE_BYTES:
            raise ParseError('source size limit')
        self.tokens, self.index, self.depth = tokenize(text), 0, 0
        self.inputs, self.outputs, self.lets = {}, {}, []
        self.output_order, self.assigned, self.env = [], {}, {}
        self.contracts, self.labels, self.rewrites = [], {}, []

    # token helpers
    def peek(self, offset=0):
        return self.tokens[min(self.index + offset, len(self.tokens) - 1)]

    def at(self, text):
        token = self.peek()
        return token.kind in ('name', 'symbol') and token.text == text

    def advance(self):
        token = self.peek()
        self.index += 1
        return token

    def expect(self, text):
        token = self.advance()
        if token.kind not in ('name', 'symbol') or token.text != text:
            found = token.text or 'end of input'
            raise ParseError(f'expected {text!r}, found {found!r}', token.position)
        return token

    def name(self, what):
        token = self.advance()
        if token.kind != 'name':
            raise ParseError(f'expected {what}, found {token.text or "end of input"!r}', token.position)
        if token.text in RESERVED:
            raise ParseError(f'{token.text!r} is reserved', token.position)
        return token

    def nest(self, position):
        self.depth += 1
        if self.depth > MAX_NESTING:
            raise ParseError('nesting limit exceeded', position)

    # module structure
    def module(self):
        self.expect('module')
        module_name = self.name('module name').text
        self.expect('{')
        while not self.at('}'):
            if self.peek().kind == 'eof':
                raise ParseError("expected '}' to close module", self.peek().position)
            self.statement()
        self.expect('}')
        if self.peek().kind != 'eof':
            raise ParseError('text after module end', self.peek().position)
        for key in self.output_order:
            if key not in self.assigned:
                raise ParseError(f'output {key!r} is never assigned', self.outputs[key]['position'])
        program = {'format': 'wspl.scalar_ir.v0',
                   'inputs': self.inputs,
                   'lets': self.lets,
                   'outputs': [{'name': k, 'type': self.outputs[k]['type'], 'value': self.assigned[k]}
                               for k in self.output_order]}
        try:
            ScalarProgram(program,max_nodes=self.max_nodes)
        except ValueError as exc:
            raise ParseError(f'IR validation failed: {exc}') from exc
        return module_name, program

    def statement(self):
        token = self.peek()
        if token.kind == 'name' and token.text in UNSUPPORTED:
            raise ParseError(f'{token.text!r} declarations are outside the Core PWA v0 parser', token.position)
        if self.at('input') or self.at('output'):
            self.port(self.advance().text)
        elif self.at('let'):
            self.let()
        elif self.at('ensures') or self.at('preserves'):
            self.contract(self.advance().text)
        elif token.kind == 'name' and self.peek(1).text == '=':
            self.assignment()
        else:
            raise ParseError(f'unexpected {token.text or "end of input"!r}', token.position)

    def declare(self, token):
        if not _PORT_NAME.fullmatch(token.text):
            raise ParseError(f'name {token.text!r} must match [a-z][a-z0-9_]*', token.position)
        if token.text in self.inputs or token.text in self.outputs or token.text in self.env:
            raise ParseError(f'duplicate declaration {token.text!r}', token.position)
        return token.text

    def port(self, role):
        token = self.name(f'{role} name')
        key = self.declare(token)
        self.expect(':')
        ty, label = self.type(role, key)
        self.expect(';')
        if role == 'input':
            self.inputs[key] = ty
            self.env[key] = ty['unit']
        else:
            self.outputs[key] = {'type': ty, 'position': token.position}
            self.output_order.append(key)

    def type(self, role, key):
        token = self.advance()
        if token.text not in ('Real', 'Truth'):
            raise ParseError(f'expected Real or Truth type, found {token.text!r}', token.position)
        bounds = None
        if self.at('['):
            self.advance()
            lower = self.constant()
            self.expect(',')
            upper = self.constant()
            self.expect(']')
            bounds = (lower, upper)
        label = None
        if self.peek().kind == 'name':
            unit_token = self.advance()
            if not _PORT_NAME.fullmatch(unit_token.text):
                raise ParseError(f'invalid unit label {unit_token.text!r}', unit_token.position)
            label = unit_token.text
            if token.text == 'Truth':
                raise ParseError('Truth is dimensionless; no unit label allowed', unit_token.position)
        if token.text == 'Truth':
            lower, upper = bounds or (Fraction(0), Fraction(1))
            return {'kind': 'Truth', 'unit': '1', 'lower': _q(lower), 'upper': _q(upper)}, None
        if bounds is None:
            raise ParseError('Real types need explicit [lower, upper] bounds', token.position)
        if label is None and role != 'let':
            raise ParseError(f'{role} {key!r} needs a unit or normalized_* label', token.position)
        if label is not None:
            self.labels[key] = label
        unit = ir_unit(label) if label is not None else None
        return {'kind': 'Real', 'unit': unit, 'lower': _q(bounds[0]), 'upper': _q(bounds[1])}, label

    def let(self):
        self.advance()
        token = self.name('let name')
        key = self.declare(token)
        self.expect(':')
        ty, _ = self.type('let', key)
        self.expect('=')
        value, unit = self.lower(self.expression(), ty['unit'])
        self.expect(';')
        if ty['unit'] is None:
            ty['unit'] = unit
        elif ty['unit'] != unit:
            raise ParseError(f'let {key!r} declared unit {ty["unit"]!r} but expression has unit {unit!r}', token.position)
        self.lets.append({'name': key, 'type': ty, 'value': value})
        self.env[key] = ty['unit']

    def assignment(self):
        token = self.advance()
        if token.text not in self.outputs:
            raise ParseError(f'{token.text!r} is not a declared output', token.position)
        if token.text in self.assigned:
            raise ParseError(f'output {token.text!r} assigned twice', token.position)
        self.expect('=')
        ty = self.outputs[token.text]['type']
        value, unit = self.lower(self.expression(), ty['unit'])
        self.expect(';')
        if unit != ty['unit']:
            raise ParseError(f'output {token.text!r} declared unit {ty["unit"]!r} but expression has unit {unit!r}', token.position)
        self.assigned[token.text] = value

    # executable expressions
    def expression(self):
        node = self.term()
        while self.at('+') or self.at('-'):
            op = self.advance()
            node = ('add' if op.text == '+' else 'sub', node, self.term(), op.position)
        return node

    def term(self):
        node = self.unary()
        while self.at('*') or self.at('/'):
            op = self.advance()
            node = ('mul' if op.text == '*' else 'div', node, self.unary(), op.position)
        return node

    def unary(self):
        token = self.peek()
        self.nest(token.position)
        try:
            if self.at('-') or self.at('+'):
                self.advance()
                operand = self.unary()
                return ('neg', operand, token.position) if token.text == '-' else operand
            return self.primary()
        finally:
            self.depth -= 1

    def primary(self):
        token = self.advance()
        if token.kind == 'number':
            return ('num', Fraction(token.text), token.position)
        if token.text == '(':
            node = self.expression()
            self.expect(')')
            return node
        if token.kind == 'name' and self.at('('):
            if token.text not in FUNCTIONS:
                raise ParseError(f'unknown or unsupported function {token.text!r}', token.position)
            self.advance()
            args = [self.expression()]
            while self.at(','):
                self.advance()
                args.append(self.expression())
            self.expect(')')
            if len(args) != FUNCTIONS[token.text]:
                raise ParseError(f'{token.text} takes {FUNCTIONS[token.text]} arguments', token.position)
            return ('call', token.text, args, token.position)
        if token.kind == 'name' and token.text not in RESERVED:
            return ('ref', token.text, token.position)
        raise ParseError(f'expected expression, found {token.text or "end of input"!r}', token.position)

    def constant(self):
        node = self.expression()
        value = self.linear(node, lambda name, position: self._no_reference(name, position))
        if not value.constant():
            raise ParseError('constant required', node[-1])
        return value.bias

    @staticmethod
    def _no_reference(name, position):
        raise ParseError(f'constant required, found reference {name!r}', position)

    def linear(self, node, resolve):
        kind, position = node[0], node[-1]
        if kind == 'num':
            return _Linear(bias=node[1])
        if kind == 'ref':
            return resolve(node[1], position)
        if kind == 'neg':
            return self.linear(node[1], resolve).scale(-1)
        if kind == 'call':
            return self.call(node, resolve)
        a, b = self.linear(node[1], resolve), self.linear(node[2], resolve)
        if kind == 'add':
            return a.plus(b, position)
        if kind == 'sub':
            return a.plus(b.scale(-1), position)
        if kind == 'mul':
            if a.constant():
                return b.scale(a.bias)
            if b.constant():
                return a.scale(b.bias)
            raise ParseError('product of two varying values requires an extension profile', position)
        if not b.constant():
            raise ParseError('division by a varying value is outside Core PWA', position)
        if b.bias == 0:
            raise ParseError('division by zero', position)
        return a.scale(1 / b.bias)

    def resolve_program(self, name, position):
        if name in self.env:
            return _Linear.atom({'op': 'ref', 'name': name}, self.env[name])
        if name in self.outputs:
            raise ParseError(f'output {name!r} cannot be referenced in v0 expressions', position)
        raise ParseError(f'unknown or forward reference {name!r}', position)

    def lower(self, node, context_unit):
        return self.linear(node, self.resolve_program).to_ir(context_unit)

    def call(self, node, resolve):
        _, function, args, position = node
        if resolve != self.resolve_program:
            raise ParseError('functions are not allowed here', position)
        if function in ('clip', 'margin'):
            value, unit = self.lower(args[0], None)
            lower = self.linear(args[1], self._no_reference)
            upper = self.linear(args[2], self._no_reference)
            if not lower.constant() or not upper.constant():
                raise ParseError(f'{function} bounds must be constants', position)
            if lower.bias > upper.bias or (function == 'margin' and lower.bias == upper.bias):
                raise ParseError(f'invalid {function} interval', position)
            ir = {'op': function, 'value': value, 'lower': _q(lower.bias), 'upper': _q(upper.bias)}
            return _Linear.atom(ir, unit if function == 'clip' else '1')
        if function == 'relu':
            value, unit = self.lower(args[0], None)
            return _Linear.atom({'op': 'relu', 'value': value}, unit)
        if function == 'not':
            return _Linear.atom({'op': 'not', 'value': self.lower(args[0], '1')[0]}, '1')
        left_linear, right_linear = (self.linear(a, resolve) for a in args)
        if function in ('min', 'max'):
            shared = left_linear.unit() or right_linear.unit()
            left, unit = left_linear.to_ir(shared)
            right, _ = right_linear.to_ir(shared)
            return _Linear.atom({'op': function, 'left': left, 'right': right}, unit)
        left, right = left_linear.to_ir('1')[0], right_linear.to_ir('1')[0]
        if function == 'implies':
            self.rewrites.append({'line': position[0], 'column': position[1],
                                  'source': 'implies(p, q)', 'ir': 'or(not(p), q)'})
            return _Linear.atom({'op': 'or', 'left': {'op': 'not', 'value': left}, 'right': right}, '1')
        return _Linear.atom({'op': function, 'left': left, 'right': right}, '1')

    # contracts
    def contract(self, kind):
        token = self.name('contract name')
        if any(c['name'] == token.text for c in self.contracts):
            raise ParseError(f'duplicate contract {token.text!r}', token.position)
        self.expect(':')
        self.expect('forall')
        self.expect('inputs')
        self.expect(':')
        formula = self.formula()
        self.expect(';')
        if formula['op'] == 'implies':
            assumptions, ensures = formula['args']
        else:
            assumptions, ensures = {'op': 'true'}, formula
        self.contracts.append({'name': token.text, 'kind': kind, 'assumptions': assumptions, 'ensures': ensures})

    def formula(self):
        self.nest(self.peek().position)
        try:
            left = self.disjunction()
            if self.at('implies'):
                self.advance()
                return {'op': 'implies', 'args': [left, self.formula()]}
            return left
        finally:
            self.depth -= 1

    def disjunction(self):
        args = [self.conjunction()]
        while self.at('or'):
            self.advance()
            args.append(self.conjunction())
        return args[0] if len(args) == 1 else {'op': 'or', 'args': args}

    def conjunction(self):
        args = [self.negation()]
        while self.at('and'):
            self.advance()
            args.append(self.negation())
        return args[0] if len(args) == 1 else {'op': 'and', 'args': args}

    def negation(self):
        if self.at('not'):
            token = self.advance()
            self.nest(token.position)
            try:
                return {'op': 'not', 'args': [self.negation()]}
            finally:
                self.depth -= 1
        return self.atom()

    def atom(self):
        if self.at('true') or self.at('false'):
            return {'op': self.advance().text}
        if self.at('('):
            saved = self.index
            try:
                syntax = self.comparison_syntax()
            except ParseError:
                self.index = saved
                self.advance()
                inner = self.formula()
                self.expect(')')
                return inner
            return self.comparison(syntax)
        return self.comparison(self.comparison_syntax())

    def comparison_syntax(self):
        left = self.expression()
        token = self.advance()
        if token.text not in COMPARISONS:
            raise ParseError(f'expected comparison operator, found {token.text or "end of input"!r}', token.position)
        return left, token, self.expression()

    def resolve_port(self, name, position):
        if name in self.inputs:
            return _Linear.atom('input.' + name, self.inputs[name]['unit'])
        if name in self.outputs:
            return _Linear.atom('output.' + name, self.outputs[name]['type']['unit'])
        if name in self.env:
            raise ParseError(f'contracts may reference input and output ports only, not let {name!r}', position)
        raise ParseError(f'unknown port {name!r}', position)

    def comparison(self, syntax):
        left_node, token, right_node = syntax
        left, right = self.linear(left_node, self.resolve_port), self.linear(right_node, self.resolve_port)
        a, b = left.unit(), right.unit()
        if a is not None and b is not None and a != b:
            raise ParseError(f'cannot compare unit {a!r} with unit {b!r}', token.position)
        unit = a or b or '1'

        def form(value):
            return {'terms': {ir: _q(c) for c, ir, _ in value.terms.values() if c != 0},
                    'bias': _q(value.bias), 'unit': unit}
        return {'op': COMPARISONS[token.text], 'left': form(left), 'right': form(right)}


def parse_module(text, *, max_nodes=1024):
    """Parse one WSPL Core PWA module; return its IR, contracts and parse report."""
    parser = _Parser(text,max_nodes=max_nodes)
    name, program = parser.module()
    report = {'format': 'wspl.parse_report.v0', 'module': name,
              'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
              'ir_sha256': hashlib.sha256(canonical_bytes(program)).hexdigest(),
              'unit_rule': UNIT_RULE, 'unit_labels': parser.labels,
              'contract_kinds': {c['name']: c['kind'] for c in parser.contracts},
              'preserves_semantics': 'ordinary I/O contract; baseline/edited relation not implemented',
              'rewrites': parser.rewrites}
    return ParsedModule(name, program, parser.contracts, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--ir-output', type=Path, required=True)
    p.add_argument('--report-output', type=Path)
    p.add_argument('--artifact', type=Path, help='compiled artifact to bind contracts to')
    p.add_argument('--contracts-output', type=Path)
    args = p.parse_args()
    if (args.artifact is None) != (args.contracts_output is None):
        p.error('--artifact and --contracts-output must be given together')
    try:
        parsed = parse_module(args.source.read_text(encoding='utf-8'))
        documents = [(args.ir_output, parsed.program)]
        if args.report_output:
            documents.append((args.report_output, parsed.report))
        if args.artifact:
            documents.append((args.contracts_output, parsed.contract_suite(ReLUArtifact.load(args.artifact))))
        for path, document in documents:
            with path.open('x', encoding='utf-8', newline='\n') as f:
                json.dump(document, f, indent=2, allow_nan=False)
                f.write('\n')
        print(json.dumps({'module': parsed.name, 'ir_output': str(args.ir_output),
                          'ir_sha256': parsed.report['ir_sha256'], 'contracts': len(parsed.contracts)}))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
