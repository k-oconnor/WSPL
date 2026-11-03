"""Read-only stored-weight/layout/numerical audit; no environment or scoring."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.csail_breakout import BreakoutPolicy,file_hash


def audit(path,expected_sha256):
    import numpy as np
    policy=BreakoutPolicy(path,expected_sha256)
    original=policy.tensor_identity()
    w=policy.weights
    rng=np.random.Generator(np.random.PCG64(1731))
    inputs=[np.zeros(600,dtype=np.float32)]+[(rng.random(600)<.05).astype(np.float32) for _ in range(31)]
    errors=[]
    def affine(values,weights,bias):
        return [math.fsum(float(a)*float(b) for a,b in zip(values,column))+float(c)
                for column,c in zip(weights.T,bias)]
    for x in inputs:
        h1=[math.tanh(v) for v in affine(x,w['actor/dense1/kernel'],w['actor/dense1/bias'])]
        h2=[math.tanh(v) for v in affine(h1,w['actor/dense2/kernel'],w['actor/dense2/bias'])]
        reference=affine(h2,w['actor/logits/kernel'],w['actor/logits/bias'])
        errors.extend(abs(float(a)-b) for a,b in zip(policy.logits(x),reference))
    before=time.monotonic()
    for _ in range(1000):policy.logits(inputs[1])
    elapsed=time.monotonic()-before
    if max(errors)>1e-4 or policy.tensor_identity()!=original or file_hash(path)!=expected_sha256:
        raise ValueError('frozen-weight numerical audit failed')
    return dict(scope='32 synthetic binary vectors; not physical/game behavior or universal numerical equivalence',
                artifact_sha256=expected_sha256,tensor_sha256=original,parameters=sum(x.size for x in w.values()),
                shapes={k:list(v.shape) for k,v in w.items()},max_abs_logit_difference=max(errors),
                comparison='NumPy FP32 forward versus independent scalar math.fsum/tanh double reference',
                tolerance=1e-4,synthetic_inferences_per_second=1000/elapsed,
                environment_steps=0,api_calls=0,training_updates=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=audit(args.weights,args.sha256)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
