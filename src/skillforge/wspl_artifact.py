"""Independent evaluator/loader for serialized rational ReLU MLP weights.

No dependency on the source interpreter or compiler. Real-graph arithmetic is
exact; optional Torch emission checks stored coefficients, not IEEE execution.
"""
import copy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re


def _q(value):
    if type(value) is int or isinstance(value, Fraction): return Fraction(value)
    if not isinstance(value, str) or not re.fullmatch(r'[+-]?\d+(?:/\d+)?', value):
        raise ValueError('canonical rational weights/inputs required')
    try: return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc: raise ValueError('invalid rational') from exc


def canonical_bytes(document):
    return json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


class ReLUArtifact:
    def __init__(self, document):
        d = copy.deepcopy(document)
        if not isinstance(d,dict) or set(d) != {'format','semantics','compiler','source_sha256','inputs','outputs','layers','allocation','source_map'}:
            raise ValueError('unexpected neural artifact fields')
        if d['format'] != 'wspl.relu_mlp.v0' or d['semantics'] != 'rational_real_graph':
            raise ValueError('unsupported neural artifact semantics')
        if not isinstance(d['compiler'],str) or not d['compiler']:raise ValueError('compiler identity required')
        if not isinstance(d['source_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',d['source_sha256']):
            raise ValueError('source identity required')
        def ports(records):
            if not isinstance(records,list) or not records:raise ValueError('nonempty named ports required')
            seen=set();out=[]
            for record in records:
                if not isinstance(record,dict) or set(record)!={'name','type'}:raise ValueError('invalid port')
                key=record['name'];ty=record['type']
                if not isinstance(key,str) or not re.fullmatch('[a-z][a-z0-9_]*',key) or key in seen:raise ValueError('invalid/duplicate port')
                if not isinstance(ty,dict) or set(ty)!={'kind','unit','lower','upper'} or ty['kind'] not in ('Real','Truth') or not isinstance(ty['unit'],str) or not ty['unit']:
                    raise ValueError('invalid port type')
                lo,hi=_q(ty['lower']),_q(ty['upper'])
                if lo>hi or (ty['kind']=='Truth' and (ty['unit']!='1' or lo<0 or hi>1)):raise ValueError('invalid port bounds')
                seen.add(key);out.append((key,lo,hi))
            return tuple(out)
        self._inputs=ports(d['inputs']);self._outputs=ports(d['outputs'])
        layers=d['layers']
        if not isinstance(layers,list) or not 1<=len(layers)<=65:raise ValueError('invalid layer count')
        width=len(self._inputs);parsed=[];parameters=0
        for i,layer in enumerate(layers):
            if not isinstance(layer,dict) or set(layer)!={'activation','weights','bias'}:raise ValueError('invalid layer fields')
            expected='linear' if i==len(layers)-1 else 'relu'
            if layer['activation']!=expected:raise ValueError('only hidden ReLU/final linear supported')
            weights=layer['weights'];bias=layer['bias']
            if not isinstance(weights,list) or not 1<=len(weights)<=4096 or not isinstance(bias,list) or len(bias)!=len(weights):
                raise ValueError('invalid layer shape')
            rows=[]
            for row in weights:
                if not isinstance(row,list) or len(row)!=width:raise ValueError('weight orientation/width differs')
                rows.append(tuple(_q(x) for x in row))
            parsed.append((expected,tuple(rows),tuple(_q(x) for x in bias)))
            parameters+=len(weights)*(width+1);width=len(weights)
            if parameters>1_000_000:raise ValueError('artifact capacity limit')
        if width!=len(self._outputs):raise ValueError('output width differs')
        a=d['allocation']
        # fresh_mlp: compiled from scalar IR; imported: frozen tensors (wspl_import), source_sha256 = bundle hash;
        # augmented: output-patch composition (wspl_compose), source_sha256 = composition-record hash.
        if (not isinstance(a,dict) or set(a)!={'mode','widths','parameters'} or a['mode'] not in ('fresh_mlp','imported','augmented') or
            a['widths']!=[len(self._inputs)]+[len(x[1]) for x in parsed] or a['parameters']!=parameters):
            raise ValueError('allocation metadata differs from stored weights')
        if not isinstance(d['source_map'],dict):raise ValueError('invalid source map')
        self._layers=tuple(parsed);self._document=d

    @property
    def input_names(self):return tuple(x[0] for x in self._inputs)

    @property
    def output_names(self):return tuple(x[0] for x in self._outputs)

    @property
    def sha256(self):return hashlib.sha256(canonical_bytes(self._document)).hexdigest()

    def to_dict(self):return copy.deepcopy(self._document)

    def evaluate(self, inputs):
        if not isinstance(inputs,dict) or set(inputs)!=set(self.input_names):raise ValueError('input record differs')
        values=[]
        for key,lo,hi in self._inputs:
            value=_q(inputs[key])
            if not lo<=value<=hi:raise ValueError('input outside artifact domain')
            values.append(value)
        for activation,weights,bias in self._layers:
            values=[sum((w*x for w,x in zip(row,values)),b) for row,b in zip(weights,bias)]
            if activation=='relu':values=[max(Fraction(0),x) for x in values]
        return dict(zip(self.output_names,values))

    def save(self,path):
        with Path(path).open('x',encoding='utf-8',newline='\n') as f:
            json.dump(self._document,f,indent=2,allow_nan=False);f.write('\n')

    @classmethod
    def load(cls,path,expected_sha256=None):
        path=Path(path)
        if path.stat().st_size>32_000_000:raise ValueError('artifact file limit')
        artifact=cls(json.loads(path.read_text(encoding='utf-8')))
        if expected_sha256 is not None and artifact.sha256!=expected_sha256:raise ValueError('artifact identity differs')
        return artifact

    def to_torch(self,dtype=None):
        """Return an ordinary nn.Sequential and an explicit unproved profile.

        Reject coefficients that would round; exact stored coefficients do not
        establish exact arithmetic or preservation on continuous input domains.
        Input records/units/domain guards remain the caller's responsibility.
        """
        import torch
        from torch import nn
        dtype=torch.float32 if dtype is None else dtype
        if dtype not in (torch.float32,torch.float64):raise ValueError('supported numerical dtype required')
        modules=[]
        with torch.random.fork_rng(devices=[]):
            for activation,weights,bias in self._layers:
                # No unverified real-to-float rounding is silently accepted.
                for value in [v for row in weights for v in row]+list(bias):
                    stored=torch.tensor(float(value),dtype=dtype).item()
                    if not math.isfinite(stored) or Fraction.from_float(stored)!=value:
                        raise ValueError('coefficient not exactly representable; an approximation profile is required')
                layer=nn.Linear(len(weights[0]),len(weights),dtype=dtype)
                with torch.no_grad():
                    layer.weight.copy_(torch.tensor([[float(x) for x in row] for row in weights],dtype=dtype))
                    layer.bias.copy_(torch.tensor([float(x) for x in bias],dtype=dtype))
                modules.append(layer)
                if activation=='relu':modules.append(nn.ReLU())
        return nn.Sequential(*modules),dict(artifact_sha256=self.sha256,dtype=str(dtype),torch_version=torch.__version__,
                                           stored_coefficients_exact=True,execution_guarantee='not_established',
                                           input_domain_enforcement='caller',input_order=list(self.input_names),output_order=list(self.output_names))
