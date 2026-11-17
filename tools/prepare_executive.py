"""Build an unapproved specification; never register or launch a run."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT.parent/'WSPL-controller-author-v1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    old=json.loads((PARENT/'experiments/source_pins/controller_author_v1.manifest.json').read_text())
    paths={Path(p) for p in old['runner_config']['resources'] if 'CONTROLLER_AUTHOR_V1.md' not in p}
    paths.add(ROOT/'docs/wspl/EXECUTIVE_CURRICULUM_V2.md')
    paths.add(Path(sys.executable))
    results=PARENT/'results/controller_author_v1'
    rows=[json.loads(l) for l in (results/'test.jsonl').read_text().splitlines()]
    library={}
    for row in rows:
        kind=row['spec']['kind']
        if kind not in library:
            path=results/('trained_'+row['id'][:16]+'.weights.json')
            library[kind]=dict(path=str(path.resolve()),spec=row['spec'],id=row['id'])
            paths.add(path)
    for name in ['test.jsonl','replies.jsonl','block_scores.json','results.json']:
        paths.add(results/name)
    prior=ROOT.parent/'WSPL-neural-dsl-v1/results/neural_executive_v1/adapter_final.pt'
    paths.add(prior)
    resources={str(p.resolve()):digest(p) for p in sorted(paths)}
    spec=dict(model_id='neural-executive-v2',source_roots=['src','experiments','tools','tests'],
        constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],
        runner=dict(module='run_executive',symbol='run',source='experiments/run_executive.py'),
        runner_config=dict(resources=resources,library=library,context=2048,epochs=2,accumulation=8,
                           lr=.0002,warmup=160,wall_guard_seconds=43200,prior_adapter=str(prior.resolve()),
                           prior_manifest_sha256='9bde732531c5d8a939c87c7b224fefd52c9d4ee81ea4dcfba39adef08c4a09ef'))
    p=ROOT/'experiments/source_pins/neural_executive_v2.spec.json'
    p.write_bytes((json.dumps(spec,sort_keys=True,indent=2)+'\n').encode())
    print(json.dumps(dict(resources=len(resources),spec=str(p),approved=False)))


if __name__=='__main__':main()
