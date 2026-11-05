"""Linux-only preparation of a pinned CPU Go replay; no games or model calls."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.go_replay import POLICIES,GoReplayStudy,sha
from source_consistency import inventory,write_json


def main():
    import pgx
    ref=ROOT.parent/'Skillforge/data/csail_relu_reference_v1/exp6_go_from_scratch/networks'
    hashes={}
    for name in POLICIES:
        folder=ref/name
        for p in sorted(folder.iterdir()):
            if p.suffix in ('.json','.npy'):hashes[p.relative_to(ref).as_posix()]=sha(p)
    study=GoReplayStudy(str(ref),hashes)
    assert all(sum(a.size for pair in w for a in pair)==1480274 for w in study.actors.values())
    checkout=Path('/home/kevin/skillforge-pgx');commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    assert commit=='87278d2d6e677fd87248c457207b59cfa42e578d'
    installed=Path(pgx.__file__).parent
    for p in (checkout/'pgx').rglob('*.py'):
        assert sha(p)==sha(installed/p.relative_to(checkout/'pgx')),p
    paths={Path(sys.executable),Path('/usr/bin/python3.12'),Path(sys.prefix)/'pyvenv.cfg'}
    for tree in (Path(sys.prefix)/'lib/python3.12/site-packages',Path('/usr/lib/python3.12')):
        paths.update(p for p in tree.rglob('*') if p.is_file() and (p.suffix in ('.py','.so') or '.so.' in p.name))
    for name in ('libc.so.6','libm.so.6','libz.so.1','libexpat.so.1','libffi.so.8','libpthread.so.0','libdl.so.2','librt.so.1','libgcc_s.so.1','ld-linux-x86-64.so.2'):
        paths.add(Path('/lib/x86_64-linux-gnu')/name)
    paths.update(ref/p for p in hashes)
    paths.add(checkout/'LICENSE')
    resources={str(p):sha(p) for p in sorted(paths)}
    for p in ('docs/wspl/GO_REPLAY_V1.md','provenance/csail_relu_reference_v1.json'):
        resources[p]=sha(ROOT/p)
    spec=dict(schema=1,model_id='go-replay-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.go_replay',source='src/skillforge/go_replay.py',symbol='GoReplayStudy'),
        constructor_kwargs=dict(reference_root=str(ref),hashes=hashes),
        runner=dict(module='run_go_replay',source='experiments/run_go_replay.py',symbol='run'),
        runner_config=dict(resources=resources,games_per_match=2000,batch_size=32,seed_base=10300000,wall_guard_seconds=7200))
    path=ROOT/'experiments/source_pins/go_replay_v1.spec.json';manifest=path.with_name('go_replay_v1.manifest.json')
    if path.exists() or manifest.exists():raise ValueError('refuse overwrite')
    write_json(path,spec);files=inventory(ROOT,spec['source_roots']);write_json(manifest,{**spec,'files':files})
    print(json.dumps(dict(status='UNAPPROVED',sha256=sha(manifest),sources=len(files),resources=len(resources),actors=len(study.actors),games=0)))


if __name__=='__main__':main()
