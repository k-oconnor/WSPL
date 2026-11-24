"""Bounded numerical Python syntax interpreted as data, never eval/exec.

This is a restricted numerical baseline, not unrestricted Python or an OS
sandbox. Inputs/outputs are numeric arrays; no file, network, import or object
introspection primitives exist. Both author arms use this common calculator.
"""
import ast
import math
import re
import numpy as np


FUNCTIONS = {'zeros', 'ones', 'eye', 'array', 'transpose', 'reshape', 'mean',
             'sum', 'sqrt', 'abs', 'maximum', 'minimum', 'clip', 'pinv', 'ridge'}
NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,63}\Z')


class Calculator:
    def __init__(self, inputs, max_elements=5_000_000, max_operations=2_000_000_000):
        self.max_elements = max_elements
        self.max_operations = max_operations
        self.operations = self.statements = self.iterations = 0
        self.env = {}
        for name, value in inputs.items():
            self.name(name)
            self.env[name] = self.bounded(value).copy()
        if sum(v.size for v in self.env.values()) > 40_000_000:
            raise ValueError('Total input storage exceeded')

    def name(self, name):
        if not NAME.fullmatch(name) or name in FUNCTIONS or name in ('np', 'range'):
            raise ValueError('Invalid or reserved variable name')

    def charge(self, work):
        self.operations += int(work)
        if self.operations > self.max_operations:
            raise ValueError('Numerical operation limit')

    def bounded(self, value):
        a = np.asarray(value)
        if a.dtype.kind not in 'fiu' or a.ndim > 2 or a.size > self.max_elements:
            raise ValueError('Bounded numeric scalar/vector/matrix required')
        a = a.astype(np.float64)
        if not np.isfinite(a).all() or np.any(np.abs(a) > 1e12):
            raise ValueError('Nonfinite or excessive numeric value')
        return a

    def integer(self, node):
        a = self.expression(node)
        if a.shape or float(a) != int(a) or abs(float(a)) > self.max_elements:
            raise ValueError('Bounded integer required')
        return int(a)

    def shape(self, value):
        a = np.asarray(value).reshape(-1)
        if len(a) not in (1, 2) or any(x < 0 or x != int(x) for x in a):
            raise ValueError('One or two nonnegative integer dimensions required')
        shape = tuple(int(x) for x in a)
        if math.prod(shape) > self.max_elements:
            raise ValueError('Allocation limit')
        return shape

    def selection(self, node):
        if isinstance(node, ast.Tuple):
            return tuple(self.selection(x) for x in node.elts)
        if isinstance(node, ast.Slice):
            parts = [None if x is None else self.integer(x) for x in (node.lower, node.upper, node.step)]
            if parts[2] == 0:
                raise ValueError('Zero slice step')
            return slice(*parts)
        return self.integer(node)

    def expression(self, node):
        self.charge(1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return self.bounded(node.value)
        if isinstance(node, ast.Name) and node.id in self.env:
            return self.env[node.id].copy()
        if isinstance(node, (ast.List, ast.Tuple)):
            return self.bounded([self.expression(x) for x in node.elts])
        if isinstance(node, ast.Attribute) and node.attr in ('T', 'shape'):
            a = self.expression(node.value)
            return self.bounded(a.T if node.attr == 'T' else a.shape)
        if isinstance(node, ast.Subscript):
            return self.bounded(self.expression(node.value)[self.selection(node.slice)])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            a = self.expression(node.operand)
            return self.bounded(-a if isinstance(node.op, ast.USub) else a)
        if isinstance(node, ast.BinOp):
            a, b = self.expression(node.left), self.expression(node.right)
            if isinstance(node.op, ast.MatMult):
                if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                    raise ValueError('Matrix multiplication dimensions')
                self.shape([a.shape[0], b.shape[1]])
                self.charge(a.shape[0]*a.shape[1]*b.shape[1])
                return self.bounded(a @ b)
            shape = np.broadcast_shapes(a.shape, b.shape)
            if math.prod(shape) > self.max_elements:
                raise ValueError('Broadcast allocation limit')
            self.charge(math.prod(shape))
            operations = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
                          ast.Div: np.divide}
            if type(node.op) not in operations:
                raise ValueError('Unsupported arithmetic operator')
            with np.errstate(all='raise'):
                return self.bounded(operations[type(node.op)](a, b))
        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError('Positional arguments only')
            name = node.func.id if isinstance(node.func, ast.Name) else None
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == 'np':
                name = node.func.attr
            if name not in FUNCTIONS:
                raise ValueError('Unknown numerical function')
            args = [self.expression(x) for x in node.args]
            return self.call(name, args)
        raise ValueError('Unsupported expression; no Python execution')

    def call(self, name, args):
        if name in ('zeros', 'ones') and len(args) == 1:
            shape = self.shape(args[0]); self.charge(math.prod(shape))
            return getattr(np, name)(shape)
        if name == 'eye' and len(args) == 1:
            n = self.shape(args[0])[0]; self.shape([n, n]); self.charge(n*n)
            return np.eye(n)
        if name == 'array' and len(args) == 1:
            return self.bounded(args[0])
        if name == 'transpose' and len(args) == 1:
            return args[0].T.copy()
        if name == 'reshape' and len(args) == 2:
            return args[0].reshape(self.shape(args[1])).copy()
        if name in ('mean', 'sum') and len(args) in (1, 2):
            axis = None
            if len(args) == 2:
                if args[1].shape or float(args[1]) not in (0, 1, -1, -2):
                    raise ValueError('Reduction axis')
                axis = int(args[1])
            self.charge(args[0].size)
            return self.bounded(getattr(np, name)(args[0], axis=axis))
        if name in ('sqrt', 'abs') and len(args) == 1:
            self.charge(args[0].size)
            with np.errstate(all='raise'):
                return self.bounded(getattr(np, name)(args[0]))
        if name in ('minimum', 'maximum', 'clip') and len(args) == (3 if name == 'clip' else 2):
            shape = np.broadcast_shapes(*(a.shape for a in args))
            if math.prod(shape) > self.max_elements:
                raise ValueError('Broadcast allocation limit')
            self.charge(math.prod(shape))
            return self.bounded(getattr(np, name)(*args))
        if name == 'pinv' and len(args) in (1, 2):
            a = args[0]
            if a.ndim != 2 or min(a.shape) > 512:
                raise ValueError('Pseudoinverse dimension limit')
            self.charge(a.size*min(a.shape))
            cutoff = 1e-6 if len(args) == 1 else float(args[1])
            if not 0 < cutoff <= 1:
                raise ValueError('Pseudoinverse cutoff')
            return self.bounded(np.linalg.pinv(a, rcond=cutoff))
        if name == 'ridge' and len(args) == 3:
            x, y, strength = args
            if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] > 512:
                raise ValueError('Ridge requires <=512 paired rows')
            if strength.shape or not 1e-12 <= float(strength) <= 1e12:
                raise ValueError('Positive ridge regularization required')
            n, d = x.shape
            self.shape([d, y.shape[1]])
            self.charge(n*n*d+n*n*n+d*n*y.shape[1])
            return self.bounded(x.T @ np.linalg.solve(x@x.T+float(strength)*np.eye(n), y))
        raise ValueError('Numerical function arity or domain error')

    def statement(self, node):
        self.statements += 1
        if self.statements > 10_000:
            raise ValueError('Statement limit')
        if isinstance(node, ast.For):
            if node.orelse or not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call) \
                    or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != 'range' \
                    or node.iter.keywords or not 1 <= len(node.iter.args) <= 3:
                raise ValueError('Bounded range loop required')
            self.name(node.target.id)
            indices = range(*(self.integer(x) for x in node.iter.args))
            if len(indices)+self.iterations > 10_000:
                raise ValueError('Loop limit')
            for i in indices:
                self.iterations += 1
                self.env[node.target.id] = np.array(float(i))
                for child in node.body:
                    self.statement(child)
            return
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            raise ValueError('Assignments and bounded loops only')
        target = node.targets[0]
        value = self.expression(node.value)
        if isinstance(target, ast.Name):
            self.name(target.id)
            self.env[target.id] = value.copy()
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            name = target.value.id
            if name not in self.env:
                raise ValueError('Assignment to unknown array')
            updated = self.env[name].copy()
            updated[self.selection(target.slice)] = value
            self.env[name] = self.bounded(updated)
        else:
            raise ValueError('Variable or array-slice target required')
        if sum(a.size for a in self.env.values()) > 40_000_000:
            raise ValueError('Total numerical workspace limit')

    def run(self, source, outputs):
        if not isinstance(source, str) or len(source.encode()) > 200_000:
            raise ValueError('Source byte limit')
        tree = ast.parse(source)
        if sum(1 for _ in ast.walk(tree)) > 20_000:
            raise ValueError('AST node limit')
        for node in tree.body:
            self.statement(node)
        if not isinstance(outputs, list) or len(outputs) > 32:
            raise ValueError('At most 32 named outputs')
        result = {}
        for name in outputs:
            self.name(name)
            if name not in self.env:
                raise ValueError('Unknown output')
            result[name] = self.bounded(self.env[name]).astype(np.float32)
        return result, dict(operations=self.operations, statements=self.statements,
                            loop_iterations=self.iterations, arithmetic='bounded_numpy_float64')
