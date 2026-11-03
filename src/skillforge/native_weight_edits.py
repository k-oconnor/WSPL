"""Data-only reader for the published MinAtar compact weight-edit language.

No Python proposal is compiled or executed. The last edits block is read;
invalid statements are recorded/skipped, following the native baseline policy.
"""
import ast
import io
import math
import re
import tokenize

from .csail_breakout import BreakoutPolicy,TENSORS

NAMES = dict(zip(('W1','b1','W2','b2','W3','b3'),TENSORS))


def statements(block):
    pending=[];depth=0
    for line in block.splitlines():
        line=line.split('#',1)[0].strip()
        if not line:continue
        pending.append(line)
        depth+=line.count('[')-line.count(']')
        if depth<=0:
            yield '\n'.join(pending)
            pending=[];depth=0
    if pending:yield '\n'.join(pending)


def number(node):
    if isinstance(node,ast.Constant) and type(node.value) in (int,float):
        value=node.value
    elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        value=number(node.operand)*(-1 if isinstance(node.op,ast.USub) else 1)
    else:raise ValueError('numeric literal required')
    if not math.isfinite(value) or abs(value)>1e6:raise ValueError('numeric literal outside native limits')
    return value


def value(node):
    if isinstance(node,ast.List):return [value(x) for x in node.elts]
    return number(node)


def index(node):
    if isinstance(node,ast.Slice):
        def part(x):
            if x is None:return None
            v=number(x)
            if type(v) is not int:raise ValueError('integer slice bound required')
            return v
        result=slice(part(node.lower),part(node.upper),part(node.step))
        if result.step==0:raise ValueError('zero slice step')
        return result
    v=number(node)
    if type(v) is not int:raise ValueError('integer index required')
    return v


def apply_statement(weights,text):
    import numpy as np
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type==tokenize.NUMBER and not re.fullmatch(r'(?:\d+(?:\.\d*)?|\.\d+)',tok.string):
            raise ValueError('plain decimal literals required')
    tree=ast.parse(text,mode='exec')
    if len(tree.body)!=1:raise ValueError('one statement per line')
    node=tree.body[0]
    if isinstance(node,ast.Assign) and len(node.targets)==1:
        target=node.targets[0];op='set'
    elif isinstance(node,ast.AugAssign) and isinstance(node.op,(ast.Add,ast.Sub,ast.Mult)):
        target=node.target;op={ast.Add:'add',ast.Sub:'subtract',ast.Mult:'multiply'}[type(node.op)]
    else:raise ValueError('only tensor assignments and +=/-=/*= permitted')
    if not isinstance(target,ast.Subscript) or not isinstance(target.value,ast.Name) or target.value.id not in NAMES:
        raise ValueError('declared tensor slice required')
    key=NAMES[target.value.id]
    selections=target.slice.elts if isinstance(target.slice,ast.Tuple) else [target.slice]
    if len(selections)!=weights[key].ndim:raise ValueError('explicit tensor rank required')
    selected=tuple(index(x) for x in selections)
    # A rejected statement never partially changes a tensor.
    updated=weights[key].copy()
    literal=value(node.value)
    with np.errstate(over='raise',invalid='raise'):
        if op=='set':updated[selected]=literal
        elif op=='add':updated[selected]+=literal
        elif op=='subtract':updated[selected]-=literal
        else:updated[selected]*=literal
    if not np.isfinite(updated).all() or (np.abs(updated)>1e6).any():raise ValueError('edited tensor outside native limits')
    weights[key]=updated


class NativeEditedPolicy(BreakoutPolicy):
    def __init__(self):
        import numpy as np
        self.weights={key:np.zeros(shape,dtype=np.float32) for key,shape in TENSORS.items()}
        self.mode='fresh_native_edits'
        self.artifact_sha256=None
        self._edited=False

    def apply_answer(self,answer):
        if self._edited:raise ValueError('one authored answer per fresh policy')
        self._edited=True
        if not isinstance(answer,str) or len(answer.encode('utf-8'))>2_000_000:raise ValueError('bounded answer text required')
        blocks=re.findall(r'```edits[^\S\r\n]*\r?\n(.*?)```',answer,flags=re.S)
        if not blocks:raise ValueError('no complete edits block')
        rows=list(statements(blocks[-1]))
        if len(rows)>4096:raise ValueError('statement limit exceeded')
        errors=[];applied=0
        for i,text in enumerate(rows):
            try:apply_statement(self.weights,text);applied+=1
            except (ValueError,TypeError,IndexError,SyntaxError,tokenize.TokenError,FloatingPointError,OverflowError) as exc:
                errors.append(dict(statement=i,text=text,reason=f'{type(exc).__name__}: {exc}'))
        for a in self.weights.values():a.flags.writeable=False
        return dict(statements=len(rows),applied=applied,skipped=len(errors),errors=errors,
                    parameters=sum(a.size for a in self.weights.values()),
                    nonzero=sum(int((a!=0).sum()) for a in self.weights.values()),tensor_sha256=self.tensor_identity(),
                    parser_policy='last complete edits block; invalid statements skipped and retained; no proposal execution')
