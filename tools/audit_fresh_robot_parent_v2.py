"""Read-only completion audit; no model forwards, fitting, or simulator calls.

Lives outside the frozen experiment checkout to preserve its source inventory.
"""
import collections
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


PIN = '3ecff9baac23eab515a586d3bb629f9123bc97f08d243dcaf75c27021c10716e'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    study = Path(sys.argv[1]).resolve()
    out = study/'results/fresh_robot_parent_v2'
    sys.path[:0] = [str(study/'experiments'), str(study/'src')]
    from source_consistency import verify
    from skillforge.robot_codec import render, decode_action
    manifest = verify(study, out/'approved_manifest.json', PIN)
    def check_resource(item):
        path, expected = item
        assert sha(path) == expected, path
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(check_resource, manifest['runner_config']['resources'].items()):
            pass
    receipt, result, frozen = [read(out/name) for name in
                               ('source_receipt.json', 'results.json', 'frozen.json')]
    assert receipt['status'] == result['status'] == 'complete'
    assert receipt['constructor_calls'] == 1
    assert receipt['manifest_sha256'] == result['manifest_sha256'] == frozen['manifest_sha256'] == PIN
    assert read(out.parent/'fresh_robot_parent_v2.exit.json')['returncode'] == 0
    assert sha(out/'adapter_final.pt') == frozen['adapter_sha256']
    assert sha(out/'training_data.jsonl') == frozen['training_data_sha256']
    training = lines(out/'training.jsonl')
    data = lines(out/'training_data.jsonl')
    schedule = read(out/'schedule.json')
    assert len(training) == len(schedule) == 384
    for i, (row, batch) in enumerate(zip(training, schedule), 1):
        assert row['step'] == i and row['targets'] == 8*i
        assert all(math.isfinite(row[k]) for k in ('nll', 'grad_norm', 'lr'))
        assert len(batch) == 8 and all(0 <= j < len(data) for j in batch)
    teachers = read(out/'teacher_episodes.json')
    episodes = read(out/'episodes.json')
    assert len(teachers) == 64 and len(episodes) == 32
    assert {r['seed'] for r in teachers} == set(range(73100000, 73100064))
    assert {r['seed'] for r in episodes} == set(range(73200000, 73200016))
    assert {r['mission'] for r in data} == set(range(64))
    predictions = lines(out/'predictions.jsonl')
    by_mission = collections.defaultdict(list)
    for row in predictions:
        by_mission[row['arm'], row['mission']].append(row)
    assert len(by_mission) == 32
    total_ticks = total_decisions = 0
    starts = {}
    for episode in teachers+episodes:
        arm = episode.get('arm', 'train')
        i = episode['mission']
        label = f'{arm}_{i:03d}'
        rpc = lines(out/(label+'.rpc')/'simulator_rpc.jsonl')
        assert sum(r.get('request', {}).get('op') == 'start' for r in rpc) == 1
        public = rpc[1]['reply']['result']
        if arm != 'train':
            encoded = json.dumps(public, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            initial = hashlib.sha256(encoded).hexdigest()
            assert initial == episode['start_sha256']
            starts[arm, i] = initial
        trace = json.loads(gzip.decompress(Path(episode['trace_path']).read_bytes()))
        physical = trace['trace']
        decisions = trace['decision_trace']
        assert len(physical) == len(trace['sensor_trace']) == episode['physical_ticks']
        assert len(decisions) == episode['decisions']
        if arm != 'train':
            saved = by_mission[arm, i]
            assert len(saved) == len(decisions) == episode['model_calls']
        goals = public['observation']['goals']
        order = public['observation']['requested_order']
        visited, violations = [], []
        stable = cursor = 0
        previous = None
        for j, decision in enumerate(decisions):
            action = json.loads(decision['source'])
            target = action.get('goal') if action['op'] == 'reach' else None
            if target != previous:
                stable = 0
            previous = target
            ticks = decision['outcome'].get('ticks', 0)
            for row in physical[cursor:cursor+ticks]:
                assert row['target'] == target
                if target in goals:
                    distance = math.dist(row['position_m'], goals[target])
                    assert math.isclose(distance, row['distance_m'], abs_tol=1e-12)
                    stable = stable+1 if distance <= .005 else 0
                    if stable >= 3 and target not in visited:
                        if target != order[len(visited)]:
                            violations.append(target)
                        visited.append(target)
                else:
                    stable = 0
            cursor += ticks
            if arm != 'train':
                row = saved[j]
                observation = dict(observation=decision['before'], documentation=public['documentation'])
                assert row['decision'] == j and row['prompt'] == render(observation)
                assert row['action'] == action == decode_action(observation, row['code'])
        assert cursor == len(physical)
        assert visited == episode['physical_visits'] and violations == episode['order_violations']
        stopped = decisions[-1]['outcome']['status'] == 'stopped'
        success = stopped and visited == order and not violations
        assert success == episode['success'] and stopped == episode['stopped']
        if arm == 'train':
            assert success
        total_ticks += len(physical)
        total_decisions += len(decisions)
    assert all(starts['base', i] == starts['ordinary', i] for i in range(16))
    successes = {a: sum(r['success'] for r in episodes if r['arm'] == a) for a in ('base', 'ordinary')}
    assert successes == result['summary']['successes'] == {'base': 0, 'ordinary': 16}
    assert result['summary']['corrections'] == 16 and result['summary']['regressions'] == 0
    import torch
    torch.set_num_threads(4)
    final = torch.load(out/'adapter_final.pt', map_location='cpu', weights_only=True)
    rolling = torch.load(out/'rolling.pt', map_location='cpu', weights_only=True)
    assert final['manifest_sha256'] == rolling['metadata']['manifest_sha256'] == PIN
    assert final['step'] == rolling['metadata']['step'] == 384
    assert final['base_hash'] == frozen['base_hash'] == rolling['metadata']['base_hash']
    assert final['adapter'].keys() == rolling['adapter'].keys()
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(final['adapter'].items()):
        assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
        assert torch.equal(tensor, rolling['adapter'][name])
        digest.update(name.encode())
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        count += tensor.numel()
    assert count == 18_464_768
    assert digest.hexdigest() == frozen['factor_sha256'] == final['adapter_sha256']
    report = dict(status='passed', manifest_sha256=PIN, sources_verified=len(manifest['files']),
        resources_verified=len(manifest['runner_config']['resources']), updates=384, training_targets=3072,
        training_rows=len(data), teacher_missions=64, paired_evaluation_missions=16,
        successes=successes, corrections=16, regressions=0, readiness_passed=True,
        trajectories_verified=96, physical_ticks_recomputed=total_ticks,
        decisions_verified=total_decisions, saved_predictions_verified=len(predictions),
        adapter_elements=count, adapter_file_sha256=frozen['adapter_sha256'],
        factor_sha256=digest.hexdigest(), elapsed_seconds=result['elapsed_seconds'],
        audit_model_forwards=0, audit_simulator_calls=0, audit_training_updates=0,
        scope='Ordinary-task gradient-trained prerequisite; no adaptation/WSPL authorship result',
        follow_on_launched=False)
    destination = Path(sys.argv[2])
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
