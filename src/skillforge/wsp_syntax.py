"""Bounded, data-only WSPL 0.2 structural parser. No proposal execution."""
from dataclasses import dataclass
import ast
import functools
import re


TOKEN = re.compile(r'(?P<space>\s+)|(?P<comment>//[^\n]*|\#[^\n]*)|(?P<string>"(?:[^"\\\n]|\\.)*")|(?P<number>0x[0-9a-fA-F]+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|(?P<name>[A-Za-z_][A-Za-z_0-9]*)|(?P<symbol>->|\+=|-=|\*=|==|!=|<=|>=|[{}\[\](),;:.=+*/@<>-])')


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int
    line: int
    column: int


class SyntaxError02(ValueError):
    pass


class Reader:
    def __init__(self, text, *, max_bytes=1_000_000, max_tokens=200_000):
        if not isinstance(text, str) or len(text.encode()) > max_bytes:
            raise SyntaxError02('source byte limit exceeded')
        self.text = text
        self.tokens = []
        pos = 0; line = 1; col = 1
        while pos < len(text):
            match = TOKEN.match(text, pos)
            if not match:
                raise SyntaxError02(f'{line}:{col}: unexpected character')
            chunk = match.group()
            if match.lastgroup not in ('space', 'comment'):
                self.tokens.append(Token(chunk, pos, match.end(), line, col))
                if len(self.tokens) > max_tokens:
                    raise SyntaxError02('source token limit exceeded')
            if '\n' in chunk:
                line += chunk.count('\n'); col = len(chunk.rsplit('\n', 1)[-1])+1
            else:
                col += len(chunk)
            pos = match.end()
        self.tokens.append(Token('<EOF>', len(text), len(text), line, col))
        self.i = 0

    @property
    def tok(self):
        return self.tokens[self.i]

    def fail(self, reason):
        raise SyntaxError02(f'{self.tok.line}:{self.tok.column}: {reason}')

    def take(self, text=None):
        tok = self.tok
        if text is not None and tok.text != text:
            self.fail(f'expected {text!r}, found {tok.text!r}')
        if tok.text == '<EOF>':
            self.fail('unexpected end of source')
        self.i += 1
        return tok.text

    def name(self):
        if not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', self.tok.text):
            self.fail('identifier required')
        return self.take()

    def until(self, endings):
        pieces = []; stack = []
        while self.tok.text != '<EOF>':
            t = self.tok.text
            if not stack and t in endings:
                break
            if t in ('(', '['):
                stack.append(')' if t == '(' else ']')
                if len(stack) > 64:
                    self.fail('expression nesting limit')
            elif t in (')', ']'):
                if not stack or stack.pop() != t:
                    self.fail('unbalanced expression delimiter')
            elif t in ('{', '}'):
                self.fail('unexpected block in expression')
            pieces.append(self.take())
        if stack or not pieces:
            self.fail('missing or incomplete expression')
        return ' '.join(pieces)

    def type(self):
        name = self.name()
        if name not in ('Real', 'Truth'):
            self.fail('only dimensionless Real and Truth types supported')
        result = dict(kind=name)
        if self.tok.text == '[':
            if name != 'Real':
                self.fail('Truth already has range [0,1]')
            self.take('['); result['lower'] = self.until({','}); self.take(',')
            result['upper'] = self.until({']'}); self.take(']')
        return result

    def ports(self):
        self.take('('); ports = {}
        while self.tok.text != ')':
            name = self.name(); self.take(':')
            if name in ports:
                self.fail('duplicate port')
            ports[name] = self.type()
            if self.tok.text != ',':
                break
            self.take(',')
        self.take(')')
        if not ports:
            self.fail('nonempty ports required')
        return ports

    def statements(self, depth=0):
        if depth > 16:
            self.fail('construction loop nesting limit')
        self.take('{'); rows = []
        while self.tok.text != '}':
            span = dict(line=self.tok.line, column=self.tok.column)
            if self.tok.text in ('replace','transplant','add_residual'):
                kind=self.take(); selected=None
                if kind=='replace':
                    selected=self.name();self.take('with')
                artifact=self.until({'as'});self.take('as');prefix=self.name()
                fields=self.graph_fields()
                if self.tok.text==';':self.take(';')
                rows.append(dict(kind=kind,region=selected,artifact=artifact,prefix=prefix,fields=fields,span=span))
            elif self.tok.text=='redirect':
                self.take();edge=self.until({'='});self.take('=');signal=self.until({';'});self.take(';')
                rows.append(dict(kind='redirect',edge=edge,signal=signal,span=span))
            elif self.tok.text=='untie':
                self.take();parameter=self.until({'as'});self.take('as');name=self.name();self.take(';')
                rows.append(dict(kind='untie',parameter=parameter,name=name,span=span))
            elif self.tok.text == 'for':
                self.take(); name = self.name(); self.take('in'); self.take('range'); self.take('(')
                args = self.until({')'}); self.take(')')
                rows.append(dict(kind='for', name=name, arguments=args, body=self.statements(depth+1), span=span))
            elif self.tok.text == 'place':
                self.take(); name = self.name(); self.take('(')
                arguments = self.until({')'}); self.take(')'); self.take('at'); self.take('{')
                placements = {}
                while self.tok.text != '}':
                    key = self.name(); self.take('=')
                    if key in placements:
                        self.fail('duplicate placement')
                    placements[key] = self.until({';'}); self.take(';')
                self.take('}'); self.take('writes'); self.take('='); self.take('replace_producers'); self.take(';')
                rows.append(dict(kind='place', name=name, arguments=arguments, placements=placements, span=span))
            else:
                text = self.until({';'}); self.take(';')
                rows.append(dict(kind='write', text=text, span=span))
        self.take('}')
        return rows

    def graph_fields(self):
        self.take('{');fields={}
        while self.tok.text!='}':
            key=self.name();self.take('=')
            if key in fields:self.fail('duplicate graph field')
            if self.tok.text=='{':
                self.take('{');value={}
                while self.tok.text!='}':
                    k=self.until({':'});self.take(':');v=self.until({',','}'})
                    if k in value:self.fail('duplicate graph mapping key')
                    value[k]=v
                    if self.tok.text!=',':break
                    self.take(',')
                self.take('}')
            else:value=self.until({';'})
            self.take(';');fields[key]=value
        self.take('}');return fields

    def description(self):
        self.take('describe'); name = self.name(); self.take('{')
        views = {}; hypotheses = []; claims = []
        while self.tok.text != '}':
            kind = self.take()
            if kind == 'view':
                key = self.name(); self.take('=')
                if key in views:
                    self.fail('duplicate view')
                views[key] = self.until({';'}); self.take(';')
            elif kind == 'hypothesis':
                import json
                try:
                    value = json.loads(self.take())
                except (ValueError, TypeError):
                    self.fail('quoted hypothesis required')
                if not isinstance(value, str):
                    self.fail('quoted hypothesis required')
                hypotheses.append(value); self.take(';')
            elif kind == 'claim':
                key = self.name(); self.take('semantics'); self.take('='); semantics = self.name(); self.take('{')
                assumptions = []; conclusions = []
                while self.tok.text != '}':
                    term = self.take()
                    if term not in ('assume', 'ensure'):
                        self.fail('claim needs assume/ensure statements')
                    (assumptions if term == 'assume' else conclusions).append(self.until({';'})); self.take(';')
                self.take('}')
                if not conclusions or key in {c['name'] for c in claims}:
                    self.fail('unique claim with at least one ensure required')
                claims.append(dict(name=key, semantics=semantics, assumptions=assumptions, ensures=conclusions))
            else:
                self.fail('descriptions contain only views, hypotheses and claims')
        self.take('}')
        return dict(target=name, views=views, hypotheses=hypotheses, claims=claims)

    def program(self):
        self.take('wspl'); self.take('0.2'); self.take(';')
        out = dict(target=None, constants=[], circuits={}, regions={}, edit=None, descriptions=[])
        names = set()
        while self.tok.text in ('target', 'const', 'circuit', 'region'):
            kind = self.take(); name = self.name()
            if name in names or name in {'parent', 'x', 'y'} or re.fullmatch(r'[Wbh]\d+', name):
                self.fail('duplicate/reserved declaration name')
            names.add(name)
            if kind=='region':
                self.take('in');target=self.name()
                out['regions'][name]=dict(target=target,fields=self.graph_fields())
            elif kind in ('target', 'const'):
                self.take('='); expr = self.until({';'}); self.take(';')
                if kind == 'target':
                    if out['target'] is not None:
                        self.fail('one target per plan')
                    out['target'] = dict(name=name, expression=expr)
                else:
                    out['constants'].append(dict(name=name, expression=expr))
            else:
                inputs = self.ports(); self.take('->'); outputs = self.ports(); self.take('{')
                if set(inputs) & set(outputs):
                    self.fail('input/output names overlap')
                assignments = []; assigned = set(inputs)
                while self.tok.text != '}':
                    key = self.name(); ty = None
                    if self.tok.text == ':':
                        self.take(':'); ty = self.type()
                    self.take('='); expr = self.until({';'}); self.take(';')
                    if key in assigned:
                        self.fail('circuit signals are single-assignment')
                    assigned.add(key); assignments.append(dict(name=key, expression=expr, type=ty))
                self.take('}')
                if not set(outputs) <= assigned:
                    self.fail('unassigned circuit output')
                out['circuits'][name] = dict(inputs=inputs, outputs=outputs, assignments=assignments)
        if out['target'] is None:
            self.fail('target declaration required')
        if self.tok.text == 'edit':
            self.take(); parent = self.name(); self.take('as'); name = self.name()
            if parent != out['target']['name'] or name in names:
                self.fail('edit needs declared parent and fresh candidate name')
            self.take('mode'); self.take('='); mode = self.name()
            if mode not in ('fixed','graph'):
                self.fail('mode must be fixed or graph')
            numeric = 'exact_to_storage';budget=None;options=set()
            while self.tok.text in ('numeric','budget'):
                option=self.take();self.take('=')
                if option in options:self.fail('duplicate edit option')
                options.add(option)
                if option=='numeric':numeric=self.name()
                else:
                    import json
                    try:budget=json.loads(self.take())
                    except ValueError:self.fail('quoted host budget name required')
                    if not isinstance(budget,str):self.fail('quoted host budget name required')
            if numeric not in ('exact_to_storage', 'native_numpy_v1'):
                self.fail('unsupported numeric profile')
            if (mode=='graph') != (budget is not None):self.fail('graph mode requires a host budget; fixed mode does not accept one')
            out['edit'] = dict(parent=parent, name=name, mode=mode, budget=budget, numeric=numeric, body=self.statements())
        while self.tok.text == 'describe':
            out['descriptions'].append(self.description())
        if self.tok.text != '<EOF>':
            self.fail('unexpected trailing source')
        valid = {out['target']['name']}
        if out['edit']:
            valid.add(out['edit']['name'])
        if any(d['target'] not in valid for d in out['descriptions']):
            self.fail('description refers to an undeclared artifact')
        return out


def expression(text, *, mode='eval'):
    # Construction loops re-run identical statement text; parse each once.
    # Callers treat the returned tree as read-only.
    return _expression(text, mode)


@functools.lru_cache(maxsize=8192)
def _expression(text, mode):
    try:
        tree = ast.parse(text, mode=mode)
    except (SyntaxError, RecursionError) as exc:
        raise ValueError('invalid bounded expression') from exc
    pending = [(tree, 0)]; count = 0
    while pending:
        node, depth = pending.pop(); count += 1
        if depth > 64 or count > 50_000:
            raise ValueError('expression size/depth limit')
        pending.extend((child, depth+1) for child in ast.iter_child_nodes(node))
    return tree


def parse(text):
    return Reader(text).program()
