"""Offline prompts and synthetic numerical/physical-sandbox fixtures only."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.reliable_json import save


def prompts():
    from skillforge.robot_collector import preflight
    from skillforge.robot_authoring_access import prompt
    old=ROOT.parent/'WSPL-robot-authoring-v2'
    data=ROOT/'data/robot_authoring_access_v1';data.mkdir(parents=True,exist_ok=True)
    for name in ('catalog.json','instructions.md'):shutil.copyfile(old/'data/robot_authoring_v1'/name,data/name)
    manifest=json.loads((old/'experiments/source_pins/robot_authoring_v2.manifest.json').read_text())
    catalog=json.loads((old/'results/robot_authoring_v2/common_inputs/catalog.json').read_text())
    packet=dict(adapter_id=catalog['adapter_id'],projection_count=len(catalog['projections']),
                element_count=catalog['element_count'],projections=list(catalog['projections']),acquisition_rows=255)
    report={}
    for arm in ('native','wspl'):
        folder=Path(tempfile.mkdtemp(prefix='prompt_',dir=data));(folder/'empty').mkdir()
        preflight(manifest['runner_config']['cli'],data/'catalog.json',data/'instructions.md',folder,
                  prompt(arm,r'C:\isolated\inputs',packet,[]),profile='supervisor')
        report[arm]=json.loads((folder/'preflight_binding.json').read_text())
    save(ROOT/'docs/wspl/robot_access_prompt_preflight.json',dict(status='passed',author_calls=0,arms=report))
    print('Both actual offline prompts passed; no hosted requests.')


def model():
    import numpy as np
    import torch
    from skillforge.authoring_sft import AuthoringSFTStudy,adapter_state,restore_adapter,frozen_hash
    from skillforge.robot_access import inspect_acquisition
    from skillforge.robot_submissions import reconstruct
    from skillforge.wspl_lora import identity
    from skillforge.author_python_workspace import PythonWorkspace
    from skillforge.fresh_robot import sha
    started=time.monotonic();data=ROOT/'data/robot_authoring_access_v1';data.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='synthetic_',dir=data))
    checkpoint=ROOT.parent/'WSPL-fresh-robot-parent-v2/results/fresh_robot_parent_v2/adapter_final.pt'
    assert sha(checkpoint)=='2cd5bd6662db5bdb39e6d341250abfca083672e2b9794c939abc145b49470276'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    study=AuthoringSFTStudy(r'C:\Users\kevin\wspl-runtime-v1\models\Qwen2.5-Coder-1.5B-Instruct')
    m=study.model;restore_adapter(m,saved['adapter']);m.eval();m.gradient_checkpointing_disable()
    before=adapter_state(m);base=frozen_hash(m);assert base==saved['base_hash']
    original_hooks={n:(tuple(z._forward_hooks),tuple(z._forward_pre_hooks)) for n,z in m.named_modules()}
    bank={k:v.numpy() for k,v in before.items()};parent_id=identity(bank)
    names=['model.layers.0.self_attn.q_proj','model.layers.13.mlp.up_proj']
    submission=folder/'wspl.json'
    save(submission,dict(parent_id=parent_id,programs={n:'B[0,0]=B[0,0]+1/8;' for n in names},bindings=None,note='Synthetic two-site fixture'))
    wspl,_=reconstruct(bank,submission,'wspl')
    native={k:v.copy() for k,v in bank.items()}
    for n in names:native[n+'.lora_B'][0,0]=np.float32(float(native[n+'.lora_B'][0,0])+.125)
    assert identity(native)==identity(wspl)
    rows=[dict(record={'ids':[1,2,3,4]},code='A'),dict(record={'ids':[5,6,7]},code='B')]
    req=dict(tool='inspect',candidate=None,rows='all',projections=names,tokens='all',weights=True)
    a=inspect_acquisition(m,study.tokenizer,rows,bank,req,folder/'parent',lambda:None)
    b=inspect_acquisition(m,study.tokenizer,rows,native,req,folder/'native',lambda:None)
    c=inspect_acquisition(m,study.tokenizer,rows,wspl,req,folder/'wspl',lambda:None)
    with np.load(folder/'native/arrays.npz') as nb,np.load(folder/'wspl/arrays.npz') as wb:
        assert set(nb.files)==set(wb.files)
        assert all(np.array_equal(nb[k].view(np.uint32),wb[k].view(np.uint32)) for k in nb.files)
    assert a['mean_target_nll']!=b['mean_target_nll']
    assert identity({k:v.numpy() for k,v in adapter_state(m).items()})==parent_id
    assert frozen_hash(m)==base
    assert original_hooks=={n:(tuple(z._forward_hooks),tuple(z._forward_pre_hooks)) for n,z in m.named_modules()}
    # The actual sandbox can read host-created diagnostics and cannot alter them.
    workspace=PythonWorkspace(folder/'sandbox',r'C:\Users\kevin\wspl-author-native-v1\runtime')
    shared=workspace.inputs/'query_01';shared.mkdir()
    for name in ('arrays.npz','diagnostics.json'):shutil.copyfile(folder/'native'/name,shared/name)
    code='import numpy as np,json\nfrom pathlib import Path\np=Path('+repr(str(shared))+')\nwith np.load(p/"arrays.npz",allow_pickle=False) as a: print("READ_OK",len(a.files))\ntry:\n (p/"diagnostics.json").write_text("changed")\n raise AssertionError("query inputs writable")\nexcept PermissionError: print("WRITE_DENIED")\n'
    sandbox=workspace.execute(code,timeout=60)
    assert sandbox['exit_code']==0 and not sandbox['timed_out']
    assert (shared/'diagnostics.json').read_bytes()==(folder/'native/diagnostics.json').read_bytes()
    report=dict(status='passed',author_calls=0,scientific_predictions=0,synthetic_rows=2,synthetic_forward_queries=3,
                full_sequence_early_and_middle_layer=True,base_weights_readable=True,multi_pair_native_wspl_equal=True,
                native_wspl_diagnostics_bitwise_equal=True,candidate_affects_full_model_nll=True,
                exact_parent_restoration=True,frozen_base=True,hooks_removed=True,real_sandbox_readonly_inputs=True,
                query_seconds=[a['seconds'],b['seconds'],c['seconds']],peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                seconds=time.monotonic()-started,folder=str(folder))
    save(ROOT/'docs/wspl/robot_access_model_preflight.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['prompt','model']);args=p.parse_args()
    {'prompt':prompts,'model':model}[args.part]()

