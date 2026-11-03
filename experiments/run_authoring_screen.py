"""Source-gated offline transcript grader; never makes proposer/API calls."""
import hashlib
import json
from pathlib import Path
import time

from skillforge.authoring_study import ARMS, PARAMETERS, WIDTHS, AuthoringTrial, digest, mechanism


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def trial_specs(study):
    if study.get('widths',list(WIDTHS)) != list(WIDTHS) or study.get('parameters',PARAMETERS) != PARAMETERS:
        raise ValueError('declared target differs from implemented fixed architecture')
    if study.get('caps_per_trial',dict(candidates=8,total_input_plus_output_tokens=65536,candidate_bytes=24000)) != dict(candidates=8,total_input_plus_output_tokens=65536,candidate_bytes=24000):
        raise ValueError('declared caps differ from implemented screen')
    seeds = study.get('task_seeds')
    if not isinstance(seeds, list) or len(seeds) != 2 or any(type(x) is not int for x in seeds) or len(set(seeds)) != 2:
        raise ValueError('two distinct frozen task seeds required')
    if study.get('draws') != 3 or study.get('arms') != list(ARMS):
        raise ValueError('screen must include all paired 2x3x6 trials')
    tasks = [mechanism(seed) for seed in seeds]
    from skillforge.authoring_study import cases, target
    if [[target(t, x) for x in cases()] for t in tasks][0] == [target(tasks[1], x) for x in cases()]:
        raise ValueError('task seeds must produce distinct truth tables')
    return [(f'task{index}_draw{draw}_{arm}', task, arm)
            for index, task in enumerate(tasks) for draw in range(3) for arm in ARMS]


class AuthoringScreen:
    """Pinned constructor. Trials stay independent; no shared proposer state here."""
    def replay(self, study, transcripts, output):
        if (study.get('status') != 'frozen' or not study.get('model_id') or
            not study.get('sampling_protocol') or not study.get('task_access_record') or
            type(study.get('monetary_cap')) not in (int, float) or not 0 <= study['monetary_cap'] < 1e9 or
            type(study.get('wall_guard_seconds')) is not int or study['wall_guard_seconds'] < 1):
            raise ValueError('freeze model, sampling, task access, money and wall guard before grading')
        specs = trial_specs(study)
        if not isinstance(transcripts, dict) or set(transcripts) != {x[0] for x in specs}:
            raise ValueError('exactly all 36 transcripts required; missing trials cannot disappear')
        started = time.monotonic()
        results = []
        total_cost = 0.
        for trial_id, task, arm in specs:
            if time.monotonic()-started > study['wall_guard_seconds']:
                raise RuntimeError('wall guard exhausted; screen incomplete')
            trial = AuthoringTrial(task, arm, trial_id=trial_id, model_id=study['model_id'])
            transcript = transcripts[trial_id]
            if set(transcript) != {'request_sha256', 'turns', 'final_candidate'} or transcript['request_sha256'] != digest(trial.request()):
                raise ValueError('transcript request identity differs')
            if not isinstance(transcript['turns'], list) or len(transcript['turns']) > 8:
                raise ValueError('at most eight logged proposer turns')
            previous_feedback = None
            for turn in transcript['turns']:
                if set(turn) != {'candidate', 'usage', 'previous_feedback_sha256'} or turn['previous_feedback_sha256'] != previous_feedback:
                    raise ValueError('turn is not bound to actual preceding permitted feedback')
                feedback = trial.submit(turn['candidate'], turn['usage'])
                previous_feedback = digest(feedback)
            result = trial.finalize(transcript['final_candidate'])
            total_cost += result['costs']['cost']
            result['study_sha256'] = digest(study)
            write(output/f'{trial_id}.json', dict(result=result, request=trial.request(),
                  ledger=trial.ledger, artifact=None if result.get('selected_candidate') not in trial.artifacts
                  else trial.artifacts[result['selected_candidate']].to_dict()))
            if total_cost > study['monetary_cap']:
                raise RuntimeError('reported monetary cap exceeded; no completed-screen claim')
            results.append(result)
        paired = {(r['trial_id'].rsplit('_',2)[0],r['arm']):int(r['qualification']) for r in results}
        blocks = sorted({r['trial_id'].rsplit('_',2)[0] for r in results})
        contrasts = {}
        for reference in ('weights_on','code_on'):
            differences = [paired[(block,'wspl_on')]-paired[(block,reference)] for block in blocks]
            contrasts['wspl_on_minus_'+reference] = dict(paired_differences=differences,
                mean=sum(differences)/len(differences),independent_task_instances=2,
                inference='descriptive only; draws nested within task; no confirmation p-value')
        summary = dict(format='wspl.authoring_screen_results.v0', status='complete',
                       study_sha256=digest(study), trials=36, results=results,
                       costs_reported=total_cost, usage_provenance='external transcript reports, not provider-billing verified',
                       principal_contrasts=contrasts,
                       arms={arm:dict(qualified=sum(r['qualification'] for r in results if r['arm']==arm), total=6)
                             for arm in ARMS},
                       inference='nonconfirmatory screen; no independent-confirmation claim',
                       evaluator_wall_seconds=time.monotonic()-started)
        write(output/'results.json', summary)
        return summary


def run(constructor, output_dir, config):
    output = Path(output_dir)
    receipt = json.loads((output/'source_receipt.json').read_text())
    if receipt['status'] != 'verified' or hashlib.sha256((output/'approved_manifest.json').read_bytes()).hexdigest() != receipt['manifest_sha256']:
        raise RuntimeError('approved-source receipt required')
    transcript_path = Path(config['transcripts_path'])
    if transcript_path.stat().st_size > 64_000_000:
        raise ValueError('transcript bundle too large')
    raw = transcript_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != config['transcripts_sha256']:
        raise ValueError('transcript bytes differ from approved configuration')
    screen = constructor()
    write(output/'study.json', config['study'])
    screen.replay(config['study'], json.loads(raw), output)


if __name__ == '__main__':
    raise SystemExit('Use the exact approved source gate; this module has no direct scientific entry point.')
