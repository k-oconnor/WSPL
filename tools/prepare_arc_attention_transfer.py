"""Prepare frozen ARC transfer; no scientific model calls or predictions."""
import json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from source_consistency import inventory,verify,write_json
from skillforge.go_replay import sha


def main():
    pre=json.loads((ROOT/'docs/wspl/arc_attention_preflight.json').read_text());assert pre['status']=='passed'
    old=json.loads((ROOT/'experiments/source_pins/arc_baseline_v1.manifest.json').read_text())
    resources=dict(old['runner_config']['resources'])
    parent=ROOT.parent/'Skillforge-attention-probe-resume-v1/results/attention_probe_resume_v1/retrieval/final.npz'
    ridge=ROOT.parent/'Skillforge-selective-key-v1/results/selective_key_v1/key/final.npz'
    curve=ROOT.parent/'Skillforge-key-learning-v1/results/key_learning_v1'
    keys={name:str(curve/folder/'key.npz') for name,folder in [('grid_key','fixed_grid_n1024_subset17'),('astra_key','astra_n1024_subset17')]}
    datasets={name:str(Path('C:/Users/kevin/skillforge-runtime/datasets/arc_baseline_v1')/(file+'.jsonl')) for name,file in [('challenge','challenge'),('easy','easy')]}
    for p in [parent,ridge,*map(Path,keys.values()),ROOT/'docs/wspl/ARC_ATTENTION_TRANSFER_V1.md',ROOT/'docs/wspl/arc_attention_preflight.json']:
        resources[str(p)]=sha(p)
    for p,h in resources.items():
        if sha(p)!=h:raise ValueError('changed resource '+p)
    spec=dict(schema=1,model_id='arc-attention-transfer-v1',source_roots=['src','experiments','tests','tools'],
              constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
              runner=dict(module='run_arc_attention_transfer',source='experiments/run_arc_attention_transfer.py',symbol='run'),
              runner_config=dict(resources=resources,parent_bank=str(parent),ridge_bank=str(ridge),curve_keys=keys,datasets=datasets,
                                 model_sha256=pre['model_sha256'],wall_guard_seconds=7200))
    rel='experiments/source_pins/arc_attention_transfer_v2'
    for suffix in ('.spec.json','.manifest.json'):
        assert not (ROOT/(rel+suffix)).exists()
    write_json(ROOT/(rel+'.spec.json'),spec)
    manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    iso=ROOT.parent/'Skillforge-arc-attention-transfer-v2';iso.mkdir(exist_ok=False)
    for name in spec['source_roots']:shutil.copytree(ROOT/name,iso/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',iso/'AGENTS.md');h=sha(ROOT/(rel+'.manifest.json'));verify(iso,iso/(rel+'.manifest.json'),h)
    result=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(iso),author_calls=0,benchmark_predictions=0)
    write_json(ROOT/'docs/wspl/arc_attention_transfer_v1_preparation.json',result);print(json.dumps(result))


if __name__=='__main__':main()
