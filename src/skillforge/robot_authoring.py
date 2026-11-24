"""Matched full-Python author sessions, with development-only feedback."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from .author_python_workspace import PythonWorkspace
from .robot_collector import collect
from .robot_submissions import stage_submission
from .reliable_json import save


BRIEF = '''Construct a persistent LoRA repair for a frozen Qwen2.5-Coder-1.5B robot
executive. The parent has learned ordinary reaching. Calibration can be changed
from the start, switch after the first waypoint, or return to nominal after the
first waypoint. Complete ordered waypoints and stop; preserve ordinary behavior.
At deployment Qwen emits exactly one full-vocabulary greedy token per decision:
A/B select the documented motors, 0..4 reach, P probes, S stops. Module labels
are randomized. The raw codec, model base, rank16/scale2 and motors are fixed.
Only the 196 original LoRA A/B pairs may change. No deployed Python, extra hook,
prompt, decoder, router, external solver or access to the author is permitted.
Every candidate is relative to the ORIGINAL parent, not the previous candidate.

You are explicitly authorized to run full Python through the delegated tool
below. It has NumPy, PyTorch/CUDA and Transformers. Files in
your work directory persist; each call starts a fresh interpreter. Inputs and
runtime are read-only, other authors/private host files/network are inaccessible.
Do not call native CLI tools. Return exactly ONE JSON object per response:
{"tool":"python","code":"import numpy as np\\nprint(np.__version__)"}
{"tool":"evaluate","path":"submission.json"}
{"tool":"finish","method":"Describe how your final edit was constructed"}

At most16 total responses, including malformed requests and finish; at most4
evaluation attempts, including rejected submissions. Python has a300second wall
limit per call,16GiB aggregate process memory and16processes. The GPU is shared
with the frozen model. Logs return at most64KiB. Each valid submission is evaluated
on the same16 development missions (4per condition). The tool reports outcomes
and copies detailed development observations into your read-only inputs. The
independent final missions are inaccessible until all six authors and controls
freeze. A valid submission becomes the last accepted candidate even if its scores
are worse. Finish or the16response cap freezes the last accepted candidate.
No accepted candidate means authoring failure; there are no automatic retries.

You receive parent.npz (all FP32 factor arrays by exact model name), features.npz,
examples.json (all labeled acquisition observations), catalog.json,
parent_development.json and robot_codec.py. Their concrete input path is below.
All acquisition labels are from a supplied public-feedback teacher, not Astra.
The same information, tool access and budgets are given to both interfaces.
No full base checkpoint or gradient-control weights are supplied.

Acquisition-only features: X is the input to the last block's down_proj;
R is the residual entering final RMSNorm; L is parent logits for
[A,B,0,1,2,3,4,P,S]; Y is the target one-hot in that order; W is those9 frozen
lm_head rows; N is the final RMSNorm scale. Diagnostics come from the original
parent. The model still has nonlinearities/RMSNorm and a full vocabulary; these
arrays are not an exact linear model of arbitrary adapter changes.

Primary scope is direct construction: analytic supervised linear algebra is
allowed and must be disclosed. Do not use iterative optimization, gradient
training or optimizer updates. Full Python permits such operations technically;
the source audit, not the sandbox, determines classification. Training-assisted
attempts remain recorded and must not be called gradient-free authorship.
Record the method in submission.note and finish.method. If WSPL only serializes
coefficients fitted by Python, say so; that alone is not WSPL synthesis evidence.
'''


def interface(arm):
    common = '''Save submission JSON in your writable work directory. Array files
must be np.savez archives of finite FP32 vectors/matrices (NPYv1), at most25M total
elements/128MB. Refer to files with relative paths. Parent identity is catalog's
adapter_id. Maximum submission JSON size2MB; method note at most4000characters.
The trusted host validates the data independently; author code is never deployed.
'''
    if arm == 'native':
        return common+'''Native submission schema:
{"parent_id":"...","arrays":"candidate.npz","note":"method"}
WSPL package access is denied in this arm. Use ordinary numerical Python;
do not reconstruct or invoke a WSPL interpreter as an indirect authoring tool.
Write the COMPLETE original factor bank, preserving every key/shape/dtype.
Use np.load(parent_path,allow_pickle=False), copy arrays, construct edits with
ordinary Python/NumPy/PyTorch, and np.savez('candidate.npz',**bank).'''
    return common+'''WSPL submission schema:
{"parent_id":"...","programs":{"model.layers.27.mlp.down_proj":"BODY"},
 "bindings":null,"note":"method"}
BODY is WSPL fixed exact_to_storage edit syntax over this pair's parent A/B.
Examples: B[:,:]=B*3/4; for i in range(4) { B[i,0]=i/4; }
Semicolons/braces required. Operations include +,-,*,/,@ and slices; functions
copy,transpose,reshape,zeros,full,eye,f32bits. No rank/topology changes or modules.
You may set bindings to a relative NPZ file of named FP32 construction constants;
identifiers must be alphanumeric/underscore, start with a letter, and not A/B.
Use names directly in BODY, e.g. B[:,:]=C;. The host declares hash-pinned constants
and wraps each body with the original pair's exact SHA256. Both the Python that
creates bindings and the WSPL body are archived. The installed wspl_lora module
provides describe,identity,source_for_pair,construct_wspl for your local tests.
For local bindings, convert FP32 uint32 bits to wsp_artifact.Bits object arrays;
use wsp_values.pack and wsp_artifact.digest to declare their exact hashes.
Only pairs in programs change. Bound construction constants are not runtime inputs.
Compiler budgets: <=1M source bytes/pair, <=64M total arithmetic operations,
100K statements/loop iterations aggregate,180seconds independent reconstruction.
Exact construction is not a proof of robot behavior.'''


def prompt(arm, input_path, catalog, transcript):
    return BRIEF+'\n'+interface(arm)+'\nInputs: '+str(input_path)+'\nCatalog summary:\n'+json.dumps(catalog)+\
           '\nVisible transcript (each call is a fresh collection with this history):\n'+json.dumps(transcript)


def run_session(arm, common_inputs, common_packet, config, out, manifest, identity,
                guard, stopped, report, evaluate_candidate, compute_charge,
                collect_fn=collect, workspace_factory=PythonWorkspace, validate_fn=None):
    out=Path(out);out.mkdir()
    runtime=config['native_runtime'] if arm=='native' else config['author_runtime']
    workspace=workspace_factory(out/'sandbox',runtime)
    for source in Path(common_inputs).iterdir():
        if source.is_file():shutil.copyfile(source,workspace.inputs/source.name)
    transcript=[];accepted=None;attempts=0;turns=0;method=None
    def validate(submission, folder):
        args=[config['learner_python'],'-I',config['submission_worker'],'--manifest',str(manifest),
              '--sha256',identity,'--parent',str(Path(common_inputs)/'parent.npz'),
              '--submission',str(submission),'--arm',arm,'--output',str(folder)]
        with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
            proc=subprocess.Popen(args,stdout=stdout,stderr=stderr,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try: code=proc.wait(timeout=config['construction_seconds'])
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],check=True,capture_output=True,
                               creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                proc.wait(timeout=15)
                raise RuntimeError('Reconstruction timeout; stop study without retry')
        if code:raise RuntimeError('Reconstruction worker failed')
        return json.loads((folder/'result.json').read_text())
    for turn in range(config['author_responses']):
        guard();turns=turn+1
        report(status='author_call',author_arm=arm,author_turn=turns)
        folder=out/f'turn_{turn:02d}';folder.mkdir()
        text,usage=collect_fn(config['cli'],config['catalog'],config['instructions'],folder/'collection',
            prompt(arm,workspace.inputs,common_packet,transcript),config['author_call_seconds'],stopped,profile='supervisor')
        save(folder/'usage.json',usage);transcript.append(dict(role='assistant',content=text))
        started=time.monotonic();finished=False
        try:
            request=json.loads(text)
            if not isinstance(request,dict):raise ValueError('One JSON object required')
            tool=request.get('tool')
            if tool=='python' and set(request)=={'tool','code'}:
                reply=workspace.execute(request['code'],timeout=config['python_call_seconds'])
            elif tool=='evaluate':
                if attempts>=config['proposal_attempts']:raise ValueError('Evaluation attempt cap exhausted')
                attempts+=1
                if set(request)!={'tool','path'}:raise ValueError('Evaluation schema requires tool,path')
                staged=stage_submission(workspace.work,request['path'],folder/'submission',arm,common_packet['adapter_id'])
                reply=(validate_fn or validate)(staged,staged.parent)
                if reply['status']=='accepted':
                    candidate=staged.parent/'candidate.npz'
                    reply['development']=evaluate_candidate(candidate,folder/'development')
                    # Expose development only, no private physical trace/state.
                    shared=workspace.inputs/f'development_{attempts:02d}';shared.mkdir()
                    for child in (folder/'development').glob('mission_*/decisions.json'):
                        shutil.copyfile(child,shared/(child.parent.name+'.json'))
                    shutil.copyfile(folder/'development/episodes.json',shared/'episodes.json')
                    reply['development_files']=str(shared)
                    accepted=str(candidate)
            elif tool=='finish' and set(request)=={'tool','method'} and isinstance(request['method'],str):
                method=request['method'][:4000];finished=True;reply=dict(status='finished')
            else:raise ValueError('Unknown tool or invalid schema')
        except (ValueError,TypeError,KeyError,IndexError,FileNotFoundError,PermissionError) as error:
            reply=dict(status='rejected',error=str(error)[:2000])
        finally:compute_charge(time.monotonic()-started)
        transcript.append(dict(role='tool',content=reply));save(out/'transcript.json',transcript)
        save(out/'status.json',dict(turns=turns,proposal_attempts=attempts,candidate=accepted))
        if finished:break
    record=dict(candidate=accepted,proposal_attempts=attempts,turns=turns,
                status='frozen' if accepted else 'authoring_failed',method=method,
                classification='requires_source_audit_no_gradient_free_claim_yet')
    save(out/'frozen.json',record)
    return accepted
