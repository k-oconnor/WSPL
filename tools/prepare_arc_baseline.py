"""Prepare isolated baseline source/resources. No scored model predictions."""
import json
import hashlib
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,write_json,verify
from skillforge.arc_eval import validate_row


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    model=Path('C:/Users/kevin/skillforge-runtime/models/SmolLM2-135M-Instruct')
    data=Path('C:/Users/kevin/skillforge-runtime/datasets/arc_baseline_v1')
    preflight=json.loads((ROOT/'docs/wspl/smollm_bridge_preflight.json').read_text())
    if preflight['status']!='passed' or not preflight['zero_adapter_logits_bit_identical']:raise ValueError('full model preflight required')
    resources={}
    files={Path(sys.executable),Path(sys.prefix)/'pyvenv.cfg'}
    for tree in (Path(sys.prefix)/'Lib/site-packages',Path(sys.base_prefix)/'Lib',Path(sys.base_prefix)/'DLLs'):
        files.update(p for p in tree.rglob('*') if p.is_file() and p.suffix in ('.py','.pyd','.dll'))
    files.update(p for p in Path(sys.base_prefix).glob('*.dll'))
    files.update(p for p in model.iterdir() if p.is_file())
    files.update(p for p in data.rglob('*') if p.is_file() and '.cache' not in p.parts)
    files.update([ROOT/'docs/wspl/ARC_BASELINE_V1.md',ROOT/'docs/wspl/smollm_bridge_preflight.json'])
    for p in sorted(files):resources[str(p)]=sha(p)
    datasets={name:str(data/(name+'.jsonl')) for name in ('challenge','easy')}
    for name,path in datasets.items():
        rows=[json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines()]
        assert len(rows)=={'challenge':299,'easy':570}[name]
        assert len({r['id'] for r in rows})==len(rows)
        for row in rows:validate_row(row)
    spec=dict(schema=1,model_id='arc-baseline-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.arc_eval',source='src/skillforge/arc_eval.py',symbol='ArcBaseline'),constructor_kwargs=dict(model_path=str(model)),
        runner=dict(module='run_arc_baseline',source='experiments/run_arc_baseline.py',symbol='run'),
        runner_config=dict(resources=resources,datasets=datasets,wall_guard_seconds=7200))
    rel='experiments/source_pins/arc_baseline_v1'
    for suffix in ('.spec.json','.manifest.json'):
        if (ROOT/(rel+suffix)).exists():raise ValueError('refuse overwrite')
    write_json(ROOT/(rel+'.spec.json'),spec)
    manifest={**spec,'files':inventory(ROOT,spec['source_roots'])};write_json(ROOT/(rel+'.manifest.json'),manifest)
    isolated=ROOT.parent/'Skillforge-arc-baseline-v1';isolated.mkdir(exist_ok=False)
    for name in ('src','experiments','tests','tools'):
        shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(ROOT/'AGENTS.md',isolated/'AGENTS.md')
    h=sha(ROOT/(rel+'.manifest.json'));verify(isolated,isolated/(rel+'.manifest.json'),h)
    report=dict(status='UNAPPROVED',sha256=h,sources=len(manifest['files']),resources=len(resources),isolated=str(isolated),benchmark_predictions=0,author_calls=0)
    write_json(ROOT/'docs/wspl/arc_baseline_v1_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
