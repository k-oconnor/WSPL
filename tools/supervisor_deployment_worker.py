"""JSON-line deployment process. MUST run inside the offline sandbox.

The evaluator supplies only public observations. Candidate code never receives
the evaluator, simulator or private scoring state in this process.
"""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root=Path(sys.argv[1]).resolve()
    sys.path.insert(0,str(root))
    declaration=json.loads((root/'submission.json').read_text())
    entry=(root/declaration['entrypoint']).resolve()
    if not entry.is_relative_to(root) or entry.suffix!='.py':raise ValueError('Invalid entrypoint')
    spec=importlib.util.spec_from_file_location('submitted_policy',entry)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    counts={'qwen_forward_calls':0}
    try:
        import torch
        def observe_forward(model,args,output):
            if model.__class__.__name__=='Qwen2ForCausalLM':counts['qwen_forward_calls']+=1
        handle=torch.nn.modules.module.register_module_forward_hook(observe_forward)
    except ImportError:handle=None
    policy=module.Policy(root,declaration)
    for line in sys.stdin:
        request=json.loads(line)
        if request['op']=='reset':policy.reset(request['public']);result=None
        elif request['op']=='act':result=policy.act(request['observation'])
        elif request['op']=='close':policy.close();print(json.dumps(dict(ok=True,result=None)),flush=True);return
        else:raise ValueError('Unknown deployment operation')
        print(json.dumps(dict(ok=True,result=result,audit=dict(counts)),allow_nan=False),flush=True)


if __name__=='__main__':main()
