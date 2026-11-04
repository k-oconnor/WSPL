"""Create an unapproved diffusion replay pin; no sampling or launch."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.diffusion_replay import DiffusionReplayStudy, MODEL_IDS, sha
from source_consistency import inventory, write_json


def main():
    spec_path=ROOT/'experiments/source_pins/diffusion_replay_v1.spec.json'
    manifest_path=spec_path.with_name('diffusion_replay_v1.manifest.json')
    if spec_path.exists() or manifest_path.exists():
        raise ValueError('refusing to overwrite a candidate')
    reference=ROOT.parent/'Skillforge/data/csail_relu_reference_v1'
    bundle=json.loads((reference/'bundle.json').read_text())
    for row in bundle['files']:
        path=reference/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('reference identity differs: '+str(path))
    folder=reference/'exp2_diffusion/networks'
    hashes={name:sha(folder/(name+'.npz')) for name in MODEL_IDS}
    study=DiffusionReplayStudy(str(folder),hashes)  # Import only, no forward/sampling.
    if sum(a.size for a in study.models[MODEL_IDS[0]].weights.values())!=23824:
        raise ValueError('parameter count differs')
    paths={Path(sys.executable),Path(sys.prefix)/'pyvenv.cfg',reference/'bundle.json',
           reference/'LICENSE',folder/'description.txt'}
    paths.update(folder/(name+'.npz') for name in MODEL_IDS)
    # Bind Python/stdlib and NumPy implementation/native BLAS. No Torch imported.
    base=Path(sys.base_prefix)
    for tree in (base/'Lib',base/'DLLs',Path(np.__file__).parent,Path(np.__file__).parent.parent/'numpy.libs'):
        if tree.is_dir():
            paths.update(p for p in tree.rglob('*') if p.is_file() and p.suffix.lower() in ('.py','.pyd','.dll','.zip') and 'site-packages' not in p.relative_to(tree).parts)
    paths.update(p for p in base.iterdir() if p.suffix.lower() in ('.dll','.exe','.zip'))
    resources={str(p.resolve()):sha(p) for p in sorted(paths)}
    resources['docs/wspl/DIFFUSION_REPLAY_V1.md']=sha(ROOT/'docs/wspl/DIFFUSION_REPLAY_V1.md')
    spec=dict(schema=1,model_id='diffusion-replay-v1',source_roots=['src','experiments','tests','tools'],
        constructor=dict(module='skillforge.diffusion_replay',source='src/skillforge/diffusion_replay.py',symbol='DiffusionReplayStudy'),
        runner=dict(module='run_diffusion_replay',source='experiments/run_diffusion_replay.py',symbol='run'),
        constructor_kwargs=dict(reference_root=str(folder),model_hashes=hashes),
        runner_config=dict(resources=resources,samples_per_model=1024,batch_size=64,seed_base=9200000,wall_guard_seconds=1800))
    write_json(spec_path,spec)
    files=inventory(ROOT,spec['source_roots'])
    write_json(manifest_path,{**spec,'files':files})
    print(json.dumps(dict(status='UNAPPROVED',manifest_sha256=sha(manifest_path),sources=len(files),resources=len(resources),numpy=np.__version__,sampled_images=0)))


if __name__=='__main__': main()
