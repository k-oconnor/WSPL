"""Bounded author sessions: zero CLI tools, numeric data requests, development only."""
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .robot_collector import collect
from .robot_construction import aliases, numerical_inputs, summaries
from .wspl_lora import describe
from .reliable_json import save


BRIEF = '''Construct a persistent robot-executive LoRA repair for a frozen Qwen model.
The parent knows ordinary reaching. Calibration may start changed, switch after
the first waypoint, or return to nominal after the first waypoint. All missions
require ordered waypoints and an explicit stop. Preserve ordinary behavior.
At deployment Qwen chooses exactly one full-vocabulary argmax token: A/B select
the documented fixed motors, 0..4 reach waypoints, P issues a fixed +x probe,
S stops. The raw-observation codec, base, adapter topology and motors are fixed.
You may construct A/B factors only. No prompt, decoder, router, hook or external
inference program can change. Rank16, scale2; all other factors retain the parent.
Every proposal starts from the ORIGINAL parent, not the prior proposal.

You have at most 16 responses and 4 proposal attempts, including rejected ones.
Return one JSON request, never call a native CLI tool. Your full visible history
is supplied on each call; hidden reasoning continuity is not promised.
Tools:
{"tool":"examples","offset":0,"count":4} returns up to8 acquisition examples.
{"tool":"inspect","tensor":"X","offset":0,"count":32} returns up to256 flat values.
{"tool":"propose","compute":{"source":"C=ridge(X,Y,0.1)","outputs":["C"]},
 "patches":{"model.layers.27.mlp.down_proj":"BODY"}}
performs common numerical construction, then interprets each factor BODY in your
assigned interface. Only outputs of compute are available to BODY besides A/B.
Host supplies exact parent/binding identities and archives the expanded source.
{"tool":"finish"} freezes your last accepted candidate. No accepted candidate
means authoring failure. Last accepted is frozen automatically at the call cap.

The shared calculator is a RESTRICTED data interpreter, not unrestricted Python.
Statements: assignments, array-slice assignments, bounded for loops over range.
Expressions: numeric literals/lists, + - * / @, slices, .T, .shape, and functions
zeros,ones,eye,array,transpose,reshape,mean,sum,sqrt,abs,maximum,minimum,clip,pinv,ridge.
Use positional arguments. mean/sum optional second argument is the axis.
pinv(M,rcond) is a numerical pseudoinverse; ridge(X,Y,lambda) returns the closed-
form coefficients X.T @ solve(X@X.T + lambda*I,Y). It is supervised fitting.
No imports, files, strings, arbitrary calls, conditionals, while loops or autograd.
Intermediate arithmetic is float64; exported bindings/factors are stored FP32.
Limits: 5M elements per intermediate, 40M total, 2B arithmetic operations,
10K statements/loop iterations, 200KB common source. Export at most32 arrays,
each <=1M elements and <=2M total. Export names may not be A/B or start with P.

Common acquisition-only arrays:
X: input to final transformer block's down_proj, [examples,intermediate_width].
R: residual entering final RMSNorm, [examples,hidden_width].
L: parent's logits for [A,B,0,1,2,3,4,P,S], [examples,9].
Y: supervised one-hot target action in that order, [examples,9].
W: frozen lm_head rows for those actions, [9,hidden_width].
N: frozen final RMSNorm learned scale, [hidden_width].
PnnnA/PnnnB: all original parent factors, according to the supplied alias catalog.
These diagnostics are from the parent. They do not provide a full exact linear
model of the transformer: nonlinearities, RMSNorm and the full-vocabulary argmax
still matter. Editing another projection is allowed but X is final-down input.

Each valid proposal is evaluated on the same16 DEVELOPMENT missions (four per
condition), with measured errors and short failure traces. Final missions remain
unavailable until all candidates freeze. Four proposal attempts is a hard cap;
do not ask for final cases or another author's results. Transport/infrastructure
failures stop the whole study without automatic retries. Invalid proposals use
the response/proposal allowance and return an error.

Analytic supervised construction is allowed. Do not implement iterative fitting
or gradient training in the calculator. Such attempts will be retained and
classified as training-assisted if discovered by the source audit. A WSPL source
that merely copies fully computed factors is serialization, not evidence that
WSPL contributed to synthesis. Explain the mechanism in your JSON 'note' field
only if requested schemas allow it; otherwise keep reasoning in source comments.
'''


def interface(arm):
    if arm == 'native':
        return '''Native BODY uses the same restricted numerical Python syntax,
with A and B initialized from this projection's parent factors. Assign to A/B
or their slices; output A/B must preserve shapes. Example: B[:,:]=B*0.75.
This arm is a restricted numerical baseline, not arbitrary Python execution.'''
    return '''WSPL BODY uses fixed exact_to_storage tensor edit syntax, with A/B
initialized from the projection's parent. Semicolons and braces are required.
Examples: B[:,:]=B*3/4; for i in range(4) { B[i,0]=i/4; }
Functions: copy,transpose,reshape,zeros,full,eye,f32bits. Matrix multiplication @.
Exported compute arrays are pinned construction constants with their output names.
Host wraps BODY in wspl0.2 target+edit declarations with exact parent/binding hashes.
No graph/topology/rank changes, modules, imports or external runtime operations.
Source semantics prove only construction identity, not robot behavior.'''


def packet(parent, features, diagnostic_name, base_dev):
    catalog = describe(parent)
    lookup = aliases(parent)
    return dict(adapter_id=catalog['adapter_id'], projections={p: dict(**v, aliases=lookup[p])
        for p, v in catalog['projections'].items()}, feature_arrays=summaries(features),
        diagnostic_projection=diagnostic_name,
        parent_development={c: dict(n=sum(r['condition']==c for r in base_dev),
            successes=sum(r['success'] for r in base_dev if r['condition']==c))
            for c in ('ordinary', 'changed', 'switch', 'return')})


def run_session(arm, parent, features, examples, common_packet, config, out,
                manifest, identity, guard, report, evaluate_candidate, compute_charge):
    out = Path(out); out.mkdir()
    transcript, accepted = [], None
    proposal_attempts = 0
    inputs = numerical_inputs(parent, features)
    brief = BRIEF+'\n'+interface(arm)+'\nCommon packet:\n'+json.dumps(common_packet)
    for turn in range(config['author_responses']):
        guard(); report(status='author_call', author_arm=arm, author_turn=turn)
        folder = out/f'turn_{turn:02d}'; folder.mkdir()
        prompt = brief+'\nVisible transcript:\n'+json.dumps(transcript)
        answer, usage = collect(config['cli'], config['catalog'], config['instructions'],
            folder/'collection', prompt, 1200, lambda: guard(return_bool=True), profile='readonly')
        save(folder/'usage.json', usage)
        transcript.append(dict(role='assistant', content=answer))
        started = time.monotonic()
        try:
            request = json.loads(answer)
            if not isinstance(request, dict):
                raise ValueError('One JSON object required')
            tool = request.get('tool')
            if tool == 'finish' and set(request) == {'tool'}:
                break
            if tool in ('inspect', 'examples'):
                expected = {'tool', 'offset', 'count'} | ({'tensor'} if tool == 'inspect' else set())
                if set(request) != expected or type(request['offset']) is not int or type(request['count']) is not int:
                    raise ValueError('Invalid inspection schema')
                lo, count = request['offset'], request['count']
                if lo < 0 or not 1 <= count <= (256 if tool == 'inspect' else 8):
                    raise ValueError('Inspection bounds')
                if tool == 'examples':
                    reply = dict(examples=examples[lo:lo+count], total=len(examples))
                else:
                    a = inputs[request['tensor']]
                    reply = dict(shape=list(a.shape), offset=lo, values=a.reshape(-1)[lo:lo+count].tolist())
            elif tool == 'propose':
                proposal_attempts += 1
                if proposal_attempts > 4:
                    raise ValueError('Proposal attempt budget exhausted')
                save(folder/'request.json', request)
                worker_out = folder/'construction'; worker_out.mkdir()
                args = [config['learner_python'], '-I', config['construction_worker'],
                    '--manifest', str(manifest), '--sha256', identity, '--parent', config['parent_npz'],
                    '--features', config['features_npz'], '--request', str(folder/'request.json'),
                    '--output', str(worker_out), '--arm', arm]
                with (worker_out/'stdout.log').open('wb') as stdout, (worker_out/'stderr.log').open('wb') as stderr:
                    process = subprocess.Popen(args, stdout=stdout, stderr=stderr,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    try:
                        code = process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                            check=True, capture_output=True,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        process.wait(timeout=15)
                        raise RuntimeError('Construction process timeout; study stops without retry')
                if code:
                    raise RuntimeError('Construction worker infrastructure failure')
                reply = json.loads((worker_out/'result.json').read_text())
                if reply['status'] == 'accepted':
                    candidate = worker_out/'candidate.npz'
                    development = evaluate_candidate(candidate, folder/'development')
                    accepted = str(candidate)
                    reply['development'] = development
            else:
                raise ValueError('Unknown tool/schema')
        except (ValueError, TypeError, KeyError, IndexError) as error:
            reply = dict(status='rejected', error=str(error)[:2000])
        finally:
            compute_charge(time.monotonic()-started)
        transcript.append(dict(role='tool', content=reply))
        save(out/'transcript.json', transcript)
        save(out/'status.json', dict(turns=turn+1, proposal_attempts=proposal_attempts, accepted=accepted))
    save(out/'frozen.json', dict(candidate=accepted, proposal_attempts=proposal_attempts,
                               turns=turn+1, status='frozen' if accepted else 'authoring_failed'))
    return accepted
