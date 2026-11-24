"""Offline prompt checks and synthetic full-size reconstruction; never a study."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.reliable_json import save


def prompt_check():
    from skillforge.robot_collector import preflight,SUPERVISOR_INSTRUCTIONS
    from skillforge.robot_authoring import prompt
    data=ROOT/'data/robot_authoring_v1'
    instructions=data/'instructions.md'
    instructions.write_text(SUPERVISOR_INSTRUCTIONS,encoding='utf8')
    cli=Path(r'C:\Users\kevin\.vscode\extensions\openai.chatgpt-26.1117.41037-win32-x64\bin\windows-x86_64\codex.exe')
    results={}
    for arm in ('native','wspl'):
        folder=Path(tempfile.mkdtemp(prefix='full_python_prompt_',dir=data));(folder/'empty').mkdir()
        message=prompt(arm,r'C:\isolated\inputs',dict(adapter_id='synthetic',projection_count=196),[])
        preflight(cli,data/'catalog.json',instructions,folder,message,profile='supervisor')
        results[arm]=dict(status='passed',folder=str(folder),binding=json.loads((folder/'preflight_binding.json').read_text()))
    save(ROOT/'docs/wspl/robot_authoring_prompt_preflight.json',dict(status='passed',author_calls=0,arms=results))
    print(json.dumps(results))


def model_check():
    import numpy as np
    import torch
    from skillforge.authoring_sft import AuthoringSFTStudy,adapter_state,restore_adapter,frozen_hash,checkpoint
    from skillforge.teaching_training import state_hash
    from skillforge.robot_submissions import reconstruct,load_arrays
    from skillforge.wspl_lora import identity
    from skillforge.fresh_robot import token_loss,sha
    started=time.monotonic()
    data=Path(tempfile.mkdtemp(prefix='full_size_fixture_',dir=ROOT/'data/robot_authoring_v1'))
    parent_path=ROOT.parent/'WSPL-fresh-robot-parent-v2/results/fresh_robot_parent_v2/adapter_final.pt'
    if sha(parent_path)!='2cd5bd6662db5bdb39e6d341250abfca083672e2b9794c939abc145b49470276':raise ValueError('Frozen parent changed')
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    state=parent['adapter'];bank={k:v.numpy().copy() for k,v in state.items()}
    original_id=identity(bank);np.savez(data/'parent.npz',**bank)
    projection='model.layers.27.mlp.down_proj'
    save(data/'wspl.json',dict(parent_id=original_id,programs={projection:'B[0,0]=B[0,0]+1/8;'},bindings=None,note='Synthetic fixture only'))
    candidate,receipt=reconstruct(bank,data/'wspl.json','wspl')
    expected={k:v.copy() for k,v in bank.items()}
    expected[projection+'.lora_B'][0,0]=np.float32(float(expected[projection+'.lora_B'][0,0])+.125)
    np.savez(data/'native.npz',**expected)
    save(data/'native.json',dict(parent_id=original_id,arrays='native.npz',note='Synthetic fixture only'))
    native,_=reconstruct(bank,data/'native.json','native')
    assert identity(candidate)==identity(native)
    assert identity(load_arrays(data/'parent.npz'))==original_id
    print('Full adapter reconstruction and NPZ validation passed',flush=True)
    study=AuthoringSFTStudy(r'C:\Users\kevin\wspl-runtime-v1\models\Qwen2.5-Coder-1.5B-Instruct')
    model=study.model;restore_adapter(model,state);model.eval();model.gradient_checkpointing_disable()
    base=frozen_hash(model);assert base==parent['base_hash']
    ids=torch.tensor([[1,2,3,4]],device='cuda')
    def forward():
        with torch.inference_mode():return model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.detach().cpu().clone()
    before=forward()
    from skillforge.robot_diagnostics import features
    synthetic_rows=[dict(record={'ids':[1,2,3,4]},code=c,condition='synthetic',prompt='Synthetic token fixture') for c in ('A','B')]
    diagnostics,_,diagnostic_name=features(model,study.tokenizer,synthetic_rows,lambda:None,lambda **k:None)
    assert diagnostics['X'].shape==(2,8960) and diagnostics['R'].shape==(2,1536)
    assert diagnostics['L'].shape==(2,9) and diagnostic_name==projection
    restore_adapter(model,{k:torch.from_numpy(v) for k,v in candidate.items()});wspl_logits=forward()
    restore_adapter(model,{k:torch.from_numpy(v) for k,v in native.items()});native_logits=forward()
    assert torch.equal(wspl_logits,native_logits) and torch.isfinite(wspl_logits).all()
    restore_adapter(model,state);assert torch.equal(before,forward())
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=3e-5,weight_decay=.01,foreach=False)
    loss=token_loss(model,dict(ids=[1,2,3,4],target=5));loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
    checkpoint(data/'synthetic-rolling.pt',model,optimizer,dict(purpose='engineering synthetic only',step=1))
    restored=torch.load(data/'synthetic-rolling.pt',map_location='cpu',weights_only=True)
    assert state_hash(restored['adapter'])==state_hash(adapter_state(model))
    optimizer.load_state_dict(restored['optimizer'])
    restore_adapter(model,state);assert state_hash(adapter_state(model))==state_hash(state)
    assert frozen_hash(model)==base
    model.eval();model.gradient_checkpointing_disable();assert torch.equal(before,forward())
    report=dict(status='passed',task_predictions=0,author_calls=0,synthetic_optimizer_updates=1,
        parent_file_sha256=sha(parent_path),parent_factor_sha256=state_hash(state),base_hash=base,
        adapter_id=original_id,elements=sum(v.size for v in bank.values()),projections=len(bank)//2,
        native_wspl_factor_parity=True,synthetic_forward_bitwise_parity=True,original_restoration=True,
        finite_full_adapter_gradients=True,optimizer_checkpoint_reload=True,frozen_base=True,
        acquisition_diagnostic_hooks_synthetic=True,diagnostic_shapes={k:list(v.shape) for k,v in diagnostics.items()},
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,folder=str(data))
    save(ROOT/'docs/wspl/robot_authoring_model_preflight.json',report);print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['prompt','model']);args=p.parse_args()
    {'prompt':prompt_check,'model':model_check}[args.part]()
