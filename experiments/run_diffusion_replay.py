"""Source-gated CPU replay, no authoring, classifier, fitting or filtering."""
import json
import os
from pathlib import Path
import time
import numpy as np
from skillforge.diffusion_replay import sha, contact_sheet_pgm


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def verify(config):
    for path, digest in config['resources'].items():
        if sha(path) != digest:
            raise ValueError('resource changed: ' + path)


class ControlledStop(Exception):
    pass


def run(constructor, output_dir, config):
    out = Path(output_dir)
    receipt = json.loads((out / 'source_receipt.json').read_text())
    if receipt['status'] != 'verified' or sha(out / 'approved_manifest.json') != receipt['manifest_sha256']:
        raise RuntimeError('approved source receipt required')
    if config['samples_per_model'] != 1024 or config['batch_size'] != 64:
        raise ValueError('fixed replay budget required')
    if np.__version__ != '1.26.4':
        raise ValueError('NumPy version differs')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError(name + ' must be 1')
    verify(config)
    study = constructor()
    started = last = time.monotonic()
    records = []
    status = 'complete'
    def callback(t):
        nonlocal last
        control = out / 'control.json'
        if time.monotonic() - started > config['wall_guard_seconds'] or (control.exists() and json.loads(control.read_text()).get('stop')):
            raise ControlledStop()
        if time.monotonic() - last >= 30:
            save(out / 'status.json', dict(status='running',completed_batches=len(records),model=model_id,batch=batch,reverse_t=t,active_seconds=time.monotonic()-started))
            last = time.monotonic()
    save(out / 'status.json', dict(status='running', completed_batches=0))
    try:
        for model_id, model in study.models.items():
            np.savez_compressed(out / (model_id + '_weights.npz'), **model.weights)
            for batch in range(16):
                # Same PCG64 stream across the three models: paired noise.
                seed = config['seed_base'] + batch
                samples, trace = study.sampler.sample(model, 64, seed, callback)
                path = out / f'{model_id}_batch_{batch:02d}.npz'
                np.savez_compressed(path, samples=samples, **trace)
                if batch == 0:
                    contact_sheet_pgm(samples, out / (model_id + '_first64.pgm'))
                records.append(dict(model=model_id,batch=batch,seed=seed,count=64,sha256=sha(path),
                    mean_pixel=float(samples.astype(np.float64).mean()),min=float(samples.min()),max=float(samples.max())))
                save(out / 'batches.json', records)
    except ControlledStop:
        status = 'paused'
    verify(config)
    summaries = {}
    for model_id in study.models:
        rows = [r for r in records if r['model'] == model_id]
        summaries[model_id] = dict(samples=64*len(rows),mean_pixel=float(np.mean([r['mean_pixel'] for r in rows])) if rows else None)
    result = dict(status=status, manifest_sha256=receipt['manifest_sha256'],summaries=summaries,
        completed_batches=len(records),active_seconds=time.monotonic()-started,
        scope='Reconstructed posterior-variance DDPM; no replicated shape accuracy or WSPL authoring claim.')
    save(out / 'results.json', result)
    save(out / 'status.json', result)
