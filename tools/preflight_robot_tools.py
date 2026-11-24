"""Real isolated Python/validation workers driven by canned synthetic replies.

No hosted author call, robot mission, Qwen prediction or scientific optimizer.
"""
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,write_json
from skillforge.robot_authoring import run_session
from skillforge.robot_submissions import load_arrays
from skillforge.wspl_lora import identity
from skillforge.fresh_robot import sha
from skillforge.reliable_json import save


def main():
    root=Path(tempfile.mkdtemp(prefix='tool_fixture_',dir=ROOT/'data/robot_authoring_v1'))
    source_report=json.loads((ROOT/'docs/wspl/robot_authoring_model_preflight.json').read_text())
    common=root/'common';common.mkdir()
    common.joinpath('parent.npz').write_bytes((Path(source_report['folder'])/'parent.npz').read_bytes())
    bank=load_arrays(common/'parent.npz');parent_id=identity(bank)
    manifest=root/'engineering.manifest.json'
    binding=dict(module='authoring_sft',symbol='unused',source='src/skillforge/authoring_sft.py')
    write_json(manifest,dict(schema=1,model_id='engineering-fixture-not-approved',source_roots=['src','experiments','tools','tests'],
                            files=inventory(ROOT,['src','experiments','tools','tests']),constructor=binding,runner=binding))
    cfg=dict(author_runtime=r'C:\Users\kevin\wspl-author-python-v1\runtime',
        native_runtime=r'C:\Users\kevin\wspl-author-native-v1\runtime',author_responses=3,proposal_attempts=1,
        author_call_seconds=1,python_call_seconds=60,construction_seconds=180,
        learner_python=r'C:\Users\kevin\wspl-runtime-v1\learner\Scripts\python.exe',
        submission_worker=str(ROOT/'tools/robot_submission_worker.py'),cli='FAKE_NO_MODEL',catalog='unused',instructions='unused')
    canary=root/'reserved-canary.txt';canary.write_text('synthetic final-only sentinel')
    started=time.monotonic();candidates={};charges=[];collections=[]
    for arm in ('native','wspl'):
        path=root/arm
        input_path=path/'sandbox/inputs/parent.npz'
        code='import json, numpy as np, torch, transformers\nfrom pathlib import Path\n'
        code+="x=torch.ones(4,device='cuda',requires_grad=True)\nx.square().sum().backward()\nassert torch.equal(x.grad,2*x)\nprint('CUDA backward passed')\n"
        code+=f"try:\n Path({str(canary)!r}).read_bytes()\nexcept PermissionError:\n print('reserved-canary denied')\nelse:\n raise RuntimeError('isolation failed')\n"
        if arm=='wspl':
            code+="import skillforge.wspl_lora\nprint('WSPL package available')\n"
            other=root/'native/sandbox/work/submission.json'
            code+=f"try:\n Path({str(other)!r}).read_bytes()\nexcept PermissionError:\n print('other-author denied')\nelse:\n raise RuntimeError('other author exposed')\n"
            submission=dict(parent_id=parent_id,programs={'model.layers.27.mlp.down_proj':'B[0,0]=B[0,0]+1/8;'},bindings=None,note='Synthetic canned fixture')
        else:
            code+="try:\n import skillforge.wspl_lora\nexcept (ImportError,PermissionError):\n print('WSPL unavailable in native arm')\nelse:\n raise RuntimeError('WSPL available in native arm')\n"
            package=Path(cfg['author_runtime'])/'Lib/site-packages/skillforge/wspl_lora.py'
            code+=f"try:\n Path({str(package)!r}).read_bytes()\nexcept PermissionError:\n print('separate WSPL runtime denied')\nelse:\n raise RuntimeError('WSPL runtime readable')\n"
            code+=f"with np.load({str(input_path)!r},allow_pickle=False) as f:\n bank={{k:f[k].copy() for k in f.files}}\n"
            code+="key='model.layers.27.mlp.down_proj.lora_B'\nbank[key][0,0]=np.float32(float(bank[key][0,0])+.125)\nnp.savez('candidate.npz',**bank)\n"
            submission=dict(parent_id=parent_id,arrays='candidate.npz',note='Synthetic canned fixture')
        code+=f"Path('submission.json').write_text(json.dumps({submission!r}))\n"
        replies=iter([dict(tool='python',code=code),dict(tool='evaluate',path='submission.json'),dict(tool='finish',method='Synthetic fixture')])
        def canned(*a,**k):
            collections.append(dict(arm=arm,profile=k['profile']));return json.dumps(next(replies)),dict(usage={},engineering_fake=True)
        def fake_evaluation(candidate,folder):
            assert (root/'native/frozen.json').exists() or arm=='native'
            load_arrays(candidate);folder.mkdir();save(folder/'episodes.json',[])
            return dict(engineering_fake=True,task_predictions=0)
        candidate=run_session(arm,common,dict(adapter_id=parent_id),cfg,path,manifest,sha(manifest),
            lambda:None,lambda:False,lambda **k:None,fake_evaluation,charges.append,collect_fn=canned)
        if candidate is None:raise ValueError('Canned valid proposal was rejected')
        transcript=json.loads((path/'transcript.json').read_text())
        if transcript[1]['content']['exit_code']!=0:raise ValueError('Python fixture failed')
        assert 'reserved-canary denied' in transcript[1]['content']['stdout']['text']
        assert 'CUDA backward passed' in transcript[1]['content']['stdout']['text']
        if arm=='wspl':
            assert 'other-author denied' in transcript[1]['content']['stdout']['text']
            assert 'WSPL package available' in transcript[1]['content']['stdout']['text']
        else:assert 'WSPL unavailable in native arm' in transcript[1]['content']['stdout']['text']
        candidates[arm]=dict(file=candidate,adapter_id=identity(load_arrays(candidate)))
    assert candidates['native']['adapter_id']==candidates['wspl']['adapter_id']
    report=dict(status='passed',folder=str(root),author_calls=0,canned_responses=len(collections),
                task_predictions=0,real_appcontainer=True,real_source_checked_validation=True,
                reserved_canary_denied=True,other_author_denied=True,native_wspl_factor_parity=True,
                native_wspl_package_denied=True,both_runtime_cuda_backward=True,wspl_package_available_to_wspl=True,
                candidates=candidates,seconds=time.monotonic()-started)
    save(ROOT/'docs/wspl/robot_authoring_tools_preflight.json',report);print(json.dumps(report))


if __name__=='__main__':main()
