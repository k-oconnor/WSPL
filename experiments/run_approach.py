"""Source-gated, unblinded approach feasibility and imitation screen.

Not a direct entry point. No API client and no claimed LLM generation trial.
"""
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import torch
from skillforge.approach import rollout, check_feasibility, audit_initialization
from skillforge.motor import CHANNELS
from skillforge.neural import MotorRMV
from skillforge.rmv import RMVReference

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    with path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_inventory(checkout):
    roots = {'libero': Path(checkout)}
    for package in ('mujoco', 'robosuite', 'torch'):
        roots[package] = Path(importlib.util.find_spec(package).origin).parent
    result = {}
    for name, root in roots.items():
        result[name] = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*'))
                        if p.is_file() and '__pycache__' not in p.parts and '.git' not in p.parts
                        and p.suffix not in ('.pyc', '.pyo')}
    return result


def fidelity(model, channels):
    if not isinstance(model, MotorRMV): return None
    reference = RMVReference(model.export())
    with torch.no_grad(): actual = model(torch.tensor(channels, dtype=torch.float32)).tolist()
    error = max(abs(a-b) for row, out in zip(channels, actual)
                for a, b in zip(out, reference.evaluate(dict(zip(CHANNELS, row)))['commands']))
    if error > 2e-6: raise ValueError(f'RMV scalar export fidelity failure: {error}')
    return error


def run(constructor, output_dir, config):
    run_started = time.monotonic()
    output = Path(output_dir)
    # Source gate creates both before calling us. A direct invocation must fail.
    receipt = json.loads((output/'source_receipt.json').read_text())
    if receipt['status'] != 'verified' or file_hash(output/'approved_manifest.json') != receipt['manifest_sha256']:
        raise RuntimeError('verified approved-source receipt required')
    identity = receipt['manifest_sha256']
    splits = [config[k] for k in ('source_seeds', 'development_seeds', 'evaluation_seeds')]
    flat = [s for split in splits for s in split]
    if any(not split for split in splits) or len(set(flat)) != len(flat):
        raise ValueError('nonempty disjoint seed pools required')
    study = constructor()
    torch.set_num_threads(2)
    bindings = json.loads((ROOT/'artifacts/local_paths.json').read_text())
    checkout = Path(bindings['libero_checkout']).resolve(strict=True)
    expected = config['environment']
    if sys.version != expected['python'] or sys.platform != expected['platform']:
        raise RuntimeError('Python runtime or platform changed')
    versions = {name: metadata.version(name) for name in expected['versions']}
    if versions != expected['versions']: raise RuntimeError('dependency versions changed')
    if external_inventory(checkout) != expected['files']:
        raise RuntimeError('external simulator/model package or LIBERO inventory changed')
    # Record audited identity before importing LIBERO. No on-run package patches.
    write(output/'environment.json', expected)
    configdir = output/'libero_config'; configdir.mkdir()
    empty = output/'empty_datasets'; empty.mkdir()
    base = checkout/'libero/libero'
    paths = dict(benchmark_root=base, bddl_files=base/'bddl_files', init_states=base/'init_files',
                 assets=base/'assets', datasets=empty.resolve())
    write(configdir/'config.yaml', {k: v.as_posix() for k, v in paths.items()})
    if 'libero.libero' in sys.modules: raise RuntimeError('fresh interpreter required')
    os.environ['LIBERO_CONFIG_PATH'] = str(configdir.resolve())
    sys.path.insert(0, str(checkout))
    from libero.libero.envs.env_wrapper import ControlEnv
    env = ControlEnv(bddl_file_name=str(base/'bddl_files/libero_goal/put_the_cream_cheese_in_the_bowl.bddl'),
                     use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False,
                     control_freq=20, horizon=config['max_steps'], ignore_done=False, hard_reset=False)
    started = run_started
    results = dict(status='running', manifest_sha256=identity, scope='unblinded pregrasp positioning engineering screen',
                   episodes=[], training={}, parameters={n: sum(p.numel() for p in m.parameters())
                                                         for n, m in study.models.items()}, api_calls=0)
    last_save = started
    episode_count = 0
    def save_models(filename, optimizers=None):
        checkpoint = dict(manifest_sha256=identity, models={n: m.state_dict() for n, m in study.models.items()},
                          optimizers=optimizers or {}, torch_rng=torch.get_rng_state(),
                          training=results['training'])
        temp = output/(filename+'.tmp'); torch.save(checkpoint, temp); temp.replace(output/filename)
    def stopped():
        control = output/'control.json'
        return (control.exists() and json.loads(control.read_text()).get('stop', False)) or time.monotonic()-started > config['wall_guard_seconds']
    def episode(method, seed, split):
        nonlocal episode_count
        row = rollout(env, study, method, seed, config['max_steps'], config['settling_steps'], config['standoff_m'])
        row['split'] = split
        write(output/f'episode_{episode_count:03d}.json', row); episode_count += 1
        results['episodes'].append({k: v for k, v in row.items() if k not in ('trace', 'preparation')})
        write(output/'status.json', results)
        if row['status'] == 'error': raise RuntimeError(row['error'])
        return row
    try:
        # Physically check task feasibility BEFORE fitting any network.
        for seed in config['development_seeds']:
            if stopped(): results['status']='stopped'; return
            ref = episode('reference', seed, 'development')
            zero = episode('zero', seed, 'development')
            if not check_feasibility(ref, zero):
                results.update(status='feasibility_failed', failed_seed=seed)
                return
        source = []
        for seed in config['source_seeds']:
            if stopped(): results['status']='stopped'; return
            row = episode('reference', seed, 'source')
            if not row['success']:
                results['status']='source_teacher_failed'; return
            source.extend((step['before_channels'], step['command']) for step in row['trace'])
        write(output/'source_pairs.json', source)
        x = torch.tensor([s[0] for s in source], dtype=torch.float32)
        y = torch.tensor([s[1] for s in source], dtype=torch.float32)
        optimizers = {}
        # Fixed-update endpoint; no best-checkpoint or final-condition selection.
        for name in ('mlp_adamw', 'rmv_adamw'):
            m = study.models[name]
            initialization_audit = audit_initialization(m, x)
            optimizer = torch.optim.AdamW(m.parameters(), lr=config['lr'], weight_decay=0.)
            losses = []
            for update in range(config['updates']):
                if stopped(): results['status']='stopped'; return
                optimizer.zero_grad(set_to_none=True)
                loss = ((m(x)-y)**2).mean()
                if not torch.isfinite(loss): raise ValueError('nonfinite imitation loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                optimizer.step(); m.project_()
                if update % 100 == 0: losses.append(dict(update=update+1, mse=float(loss.detach())))
                results['training'][name] = dict(updates=update+1, source_rows=len(source),
                                                 row_presentations=(update+1)*len(source), loss=float(loss.detach()),
                                                 loss_samples=losses, initialization_audit=initialization_audit)
                optimizers[name] = optimizer.state_dict()
                if time.monotonic()-last_save >= 1800:
                    save_models('rolling.pt', optimizers); last_save=time.monotonic()
            results['training'][name]['rmv_export_max_error'] = fidelity(m, x.tolist())
        for name, m in study.models.items():
            if isinstance(m, MotorRMV): write(output/f'{name}.rmv.json', m.export())
        for seed in config['evaluation_seeds']:
            paired = []
            for name in ('zero', 'reference', 'mlp_authored', 'rmv_authored', 'mlp_adamw', 'rmv_adamw'):
                if stopped(): results['status']='stopped'; return
                row = episode(name, seed, 'evaluation'); paired.append(row['start_sha256'])
                if name in study.models:
                    fidelity(study.models[name], [s['before_channels'] for s in row['trace']])
            if len(set(paired)) != 1: raise ValueError('paired evaluation starts differ')
        results['status'] = 'complete'
    except Exception as exc:
        results.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        results['wall_seconds'] = time.monotonic()-started
        results['total_preparation_calls'] = sum(e['preparation_calls'] for e in results['episodes'])
        results['total_motor_calls'] = sum(e['motor_calls'] for e in results['episodes'])
        results['total_reset_calls'] = sum(e['reset_calls'] for e in results['episodes'])
        results['evaluation_summary'] = {}
        for method in ('zero', 'reference', 'mlp_authored', 'rmv_authored', 'mlp_adamw', 'rmv_adamw'):
            panel = [e for e in results['episodes'] if e['split'] == 'evaluation' and e['method'] == method]
            results['evaluation_summary'][method] = dict(attempts=len(panel), successes=sum(e['success'] for e in panel),
                                                        errors=sum(e['status']=='error' for e in panel))
        write(output/'results.json', results)
        save_models('final.pt', locals().get('optimizers'))
        env.close()
        unchanged = external_inventory(checkout) == expected['files']
        results['environment_unchanged'] = unchanged
        if not unchanged: results['status']='failed_environment_changed'
        write(output/'results.json', results)
        if not unchanged: raise RuntimeError('external environment changed during run')


if __name__ == '__main__':
    from source_consistency import require_registered_runner
    require_registered_runner(__file__)
