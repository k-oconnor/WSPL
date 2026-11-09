"""Prepare bounded repair assets and synthetic/offline checks; no study launch."""
import json,sys,hashlib,shutil,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import numpy as np
import torch
from tokenizers import Tokenizer
from source_consistency import inventory,verify,write_json
from skillforge.go_replay import sha
from skillforge.grid_movement import CopyFoundation
from skillforge.arc_eval import model_hash
from skillforge.capability_bridge import modules,attached,validate_plan
from skillforge.integration_repair import scaled
from skillforge.codex_one_shot import preflight
from skillforge.diffusion_edit import linux_path
from run_capability6 import construct
from run_integration_repair import author_prompt


def main():
    old=json.loads((ROOT/'experiments/source_pins/capability6_v2.manifest.json').read_text());c=old['runner_config']
    kw=old['constructor_kwargs'];resources=dict(c['resources'])
    parent=ROOT.parent/'Skillforge-capability6-v2/results/capability6_v1'
    adapter=parent/'wspl_0/final.json';assert sha(adapter)==json.loads((parent/'frozen.json').read_text())['wspl_0']
    dev=json.loads(Path(c['dev']).read_text());test=json.loads(Path(c['test']).read_text())
    acq=dict(grids=dict(move=dev['move'][:32],copy6=dev['copy6'][:16]),prose=json.loads((ROOT/'data/prose_degradation_v1/sample.json').read_text())[:32])
    source=ROOT.parent/'rbl_rmv_icu/data/atomic_broad_50k_v1';meta=json.loads((source/'edu/dataset.json').read_text())
    assert sha(source/'tokenizer.json')==meta['tokenizer_sha256']
    for name in ('validation.bin','validation_documents.jsonl'):assert sha(source/'edu'/name)==meta['files'][name]
    tok=Tokenizer.from_file(str(source/'tokenizer.json'));tokens=np.memmap(source/'edu/validation.bin',dtype=np.uint16,mode='r')
    prose=[]
    for i,line in enumerate((source/'edu/validation_documents.jsonl').read_text().splitlines()[64:128],64):
        r=json.loads(line);ids=tokens[r['offset']:r['offset']+r['tokens']].tolist()
        assert tok.id_to_token(ids[0])=='[BOS]' and tok.id_to_token(ids[-1])=='[EOS]'
        text=tok.decode(ids[1:-1],skip_special_tokens=False);assert hashlib.sha256(text.encode()).hexdigest()==r['source_sha256']
        prose.append(dict(id=i,source_sha256=r['source_sha256'],text=text))
    reserved=dict(grids={k:test[k][:n] for k,n in [('novel',64),('larger',64),('copy6',32)]},prose=prose)
    a={tuple(r['input']) for rows in acq['grids'].values() for r in rows};b={tuple(r['input']) for rows in reserved['grids'].values() for r in rows};assert not a&b
    assert not {r['source_sha256'] for r in acq['prose']}&{r['source_sha256'] for r in prose}
    folder=ROOT/'data/integration_repair_v1';folder.mkdir(exist_ok=False)
    for name,value in [('acquisition',acq),('reserved',reserved)]:write_json(folder/(name+'.json'),value)
    guide=(ROOT/'docs/wspl/INTEGRATION_REPAIR_INTERFACE_V1.md').read_text()+'\nWSPL GRAPH REFERENCE:\n'+(ROOT/'docs/wspl/WSP_GRAPH_SURGERY.md').read_text()
    (folder/'guide.md').write_text(guide)
    study=CopyFoundation(**kw);before=model_hash(study.model);plan=json.loads(adapter.read_text())
    for alpha in (0.,.25,.5,.75):validate_plan(scaled(plan,alpha))
    ids=torch.tensor([[100,101,102]*171],device='cuda')
    with torch.inference_mode():expected=study.model(input_ids=ids,use_cache=False).logits
    bank=modules(scaled(plan,0)).eval().requires_grad_(False)
    with attached(study.model,bank),torch.inference_mode():assert torch.equal(expected,study.model(input_ids=ids,use_cache=False).logits)
    del expected,bank
    assert model_hash(study.model)==before
    subprocess.run(['wsl','-d','Ubuntu','--','/usr/bin/python3',linux_path(ROOT/c['runtime_verify']),'--verify',linux_path(c['linux_identity'])],check=True,capture_output=True,timeout=90)
    with tempfile.TemporaryDirectory(prefix='integration-check-') as temp:
        # Use only one actual trained branch for construction interface validation.
        small=dict(recipe=plan['recipe'],branches={'0_q':plan['branches']['0_q']})
        code="wspl('0_q','wspl 0.2; target base=artifact(\"base\",sha256=\"$PARENT\"); edit base as candidate mode=fixed { b[:]=0; }')"
        proposed,result=construct(code,small,{},'wspl',Path(temp),dict(worker=str(ROOT/c['worker']),python_runtime=c['python_runtime']))
        assert proposed is not None,result
        assert any(e['kind']=='wspl' for e in result['events'])
    cli=Path(json.loads(Path(c['local_paths']).read_text())['codex_cli']);pf=folder/'prompt_preflight';pf.mkdir();(pf/'empty').mkdir()
    preflight(cli,c['catalog'],c['instructions'],pf,author_prompt(guide,{'synthetic_offline_preflight':True},[],0))
    pre=dict(status='passed',synthetic_only=True,zero_strength_logits_exact=True,base_unchanged=True,sandbox_wspl_passed=True,
        scales_valid=True,disjoint_inputs=True,peak_bytes=torch.cuda.max_memory_allocated(),author_calls=0,benchmark_predictions=0)
    write_json(ROOT/'docs/wspl/integration_repair_preflight.json',pre)
    bound=[adapter,parent/'frozen.json',folder/'acquisition.json',folder/'reserved.json',folder/'guide.md',ROOT/'docs/wspl/INTEGRATION_REPAIR_V1.md',ROOT/'docs/wspl/INTEGRATION_REPAIR_INTERFACE_V1.md',ROOT/'docs/wspl/integration_repair_preflight.json',ROOT/'data/prose_degradation_v1/sample.json',source/'tokenizer.json',source/'edu/dataset.json',source/'edu/validation.bin',source/'edu/validation_documents.jsonl']
    for p in bound:resources[str(p)]=sha(p)
    for p,h in resources.items():assert sha(p)==h,p
    config={k:c[k] for k in ('local_paths','catalog','instructions','worker','python_runtime','runtime_verify','linux_identity')}
    config.update(resources=resources,adapter=str(adapter),guide=str(folder/'guide.md'),acquisition=str(folder/'acquisition.json'),reserved=str(folder/'reserved.json'),wall_guard_seconds=14400)
    spec=dict(schema=1,model_id='integration-repair-v1',source_roots=['src','experiments','tests','tools'],constructor=old['constructor'],constructor_kwargs=kw,
        runner=dict(module='run_integration_repair',source='experiments/run_integration_repair.py',symbol='run'),runner_config=config)
    rel='experiments/source_pins/integration_repair_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-integration-repair-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(iso),author_calls=0,benchmark_predictions=0)
    write_json(ROOT/'docs/wspl/integration_repair_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
