"""Freeze a small prose sample and run synthetic-only interface checks."""
import hashlib,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import numpy as np
import torch
from tokenizers import Tokenizer
from source_consistency import inventory,verify,write_json
from skillforge.go_replay import sha
from skillforge.grid_movement import CopyFoundation
from skillforge.arc_eval import model_hash
from skillforge.capability_bridge import modules,attached
from run_prose_degradation import next_token_loss


def main():
    source=ROOT.parent/'rbl_rmv_icu/data/atomic_broad_50k_v1'
    meta=json.loads((source/'edu/dataset.json').read_text())
    for name in ('validation.bin','validation_documents.jsonl'):
        assert sha(source/'edu'/name)==meta['files'][name]
    assert sha(source/'tokenizer.json')==meta['tokenizer_sha256']
    tokenizer=Tokenizer.from_file(str(source/'tokenizer.json'));tokens=np.memmap(source/'edu/validation.bin',dtype=np.uint16,mode='r')
    documents=[json.loads(line) for line in (source/'edu/validation_documents.jsonl').read_text().splitlines()[:64]]
    rows=[]
    for i,r in enumerate(documents):
        ids=tokens[r['offset']:r['offset']+r['tokens']].tolist()
        assert tokenizer.id_to_token(ids[0])=='[BOS]' and tokenizer.id_to_token(ids[-1])=='[EOS]'
        text=tokenizer.decode(ids[1:-1],skip_special_tokens=False)
        assert hashlib.sha256(text.encode()).hexdigest()==r['source_sha256']
        rows.append(dict(id=i,source_sha256=r['source_sha256'],text=text))
    folder=ROOT/'data/prose_degradation_v1';folder.mkdir(exist_ok=False);write_json(folder/'sample.json',rows)
    old=json.loads((ROOT/'experiments/source_pins/capability_arc_v1.manifest.json').read_text())
    kw=old['runner_config']['foundation'];adapter=old['runner_config']['adapters']['wspl_0']
    assert sha(adapter)==old['runner_config']['resources'][adapter]
    study=CopyFoundation(**kw);before=model_hash(study.model);bank=modules(json.loads(Path(adapter).read_text())).eval().requires_grad_(False)
    ids=torch.tensor([[100,101,102]*171],device='cuda')
    assert next_token_loss(study.model,ids)[1]==512
    with attached(study.model,bank):assert next_token_loss(study.model,ids)[1]==512
    assert model_hash(study.model)==before
    check=dict(synthetic_only=True,full_length_scoring_passed=True,peak_bytes=torch.cuda.max_memory_allocated(),documents_roundtrip_verified=64)
    write_json(ROOT/'docs/wspl/prose_degradation_preflight.json',check)
    base=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text())
    resources={p:h for p,h in base['runner_config']['resources'].items() if not Path(p).is_relative_to(ROOT)}
    for p in [folder/'sample.json',Path(adapter),Path(kw['checkpoint_path']),ROOT/'docs/wspl/PROSE_DEGRADATION_V1.md',ROOT/'docs/wspl/prose_degradation_preflight.json',source/'tokenizer.json',source/'edu/dataset.json',source/'edu/validation.bin',source/'edu/validation_documents.jsonl']:
        resources[str(p)]=sha(p)
    for p,h in resources.items():assert sha(p)==h,p
    spec=dict(schema=1,model_id='prose-degradation-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.grid_movement',source='src/skillforge/grid_movement.py',symbol='CopyFoundation'),constructor_kwargs=kw,
        runner=dict(module='run_prose_degradation',source='experiments/run_prose_degradation.py',symbol='run'),
        runner_config=dict(resources=resources,adapter=adapter,data=str(folder/'sample.json'),wall_guard_seconds=1200,
            prompts=['The water cycle describes how','To make a cup of tea, first','The purpose of a public library is']))
    rel='experiments/source_pins/prose_degradation_v1'
    for suffix in ('.spec.json','.manifest.json'):assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec);manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-prose-degradation-v1';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(iso),benchmark_predictions=0,author_calls=0)
    write_json(ROOT/'docs/wspl/prose_degradation_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
