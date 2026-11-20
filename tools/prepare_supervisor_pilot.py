"""Prepare unapproved supervisor spec and offline prompt; no scientific execution."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.supervisor_workspace import digest,save
from skillforge.supervisor_session import TOOL_INSTRUCTIONS
from skillforge.codex_one_shot import preflight


def main():
    data=ROOT/'data/supervisor_preparation';data.mkdir(parents=True,exist_ok=True)
    historical=ROOT.parent/'WSPL-executive-teaching-repair-v2/results/executive_teaching_v2'
    old=json.loads((historical/'approved_manifest.json').read_text())['runner_config']
    linux=json.loads((ROOT/'data/supervisor_linux_runtime.json').read_text())
    resources={p:h for p,h in old['resources'].items() if 'venv-sim' in p.lower() or 'pythonsoftwarefoundation.python.3.11_' in p.lower()}
    for p,h in resources.items():
        if digest(p)!=h:raise ValueError('Host runtime changed '+p)
    # Check Linux aliases/case natively; Windows cannot reliably open all of them.
    linux_manifest=ROOT/'data/supervisor_linux_runtime.json'
    resources[str(linux_manifest)]=digest(linux_manifest)
    catalog=data/'catalog.json';catalog.write_bytes(Path(old['catalog']).read_bytes())
    instructions=data/'instructions.md';instructions.write_text("Answer the user's request.\n")
    model=Path('C:/Users/kevin/skillforge-runtime/models/Qwen2.5-Coder-1.5B-Instruct')
    readonly=dict(model=str(model),parent=str(historical/'rehearsal/adapter_final.pt'),
        original=str(ROOT.parent/'WSPL-closed-loop-v1/results/compact_closed_loop_v1/seed_113/adapter_final.pt'),
        taught=str(historical/'taught/adapter_final.pt'),code=str(ROOT/'src'),history=str(historical),
        motor=str(ROOT.parent/'WSPL-factory-v1/results/factory_pilot_v1/artifact_final.json'))
    for k,path in readonly.items():
        if k=='code':continue # Inventoried Python source; never mount the repository root.
        path=Path(path)
        for p in (path.rglob('*') if path.is_dir() else [path]):
            if p.is_file():resources[str(p.resolve())]=digest(p)
    brief=ROOT/'docs/wspl/ASTRA_SUPERVISOR_BRIEF.md'
    for p in (brief,ROOT/'docs/wspl/ASTRA_SUPERVISOR_V1.md',ROOT/'docs/wspl/ASTRA_SUPERVISOR_V2_REPAIR.md',Path(old['cli']),catalog,instructions,Path(sys.executable)):
        resources[str(p.resolve())]=digest(p)
    help_text='''Development budget: 64 Astra responses, 8 cumulative GPU-job hours,
20 wall hours for development; final evaluation has a separate 4-hour guard.
Use /work for all writable files. Inspect /work/resources.json and
/inputs/history for prior failures, demonstrations, and training logs.
import sys; sys.path.insert(0, '/inputs/code')
from skillforge.supervisor_workbench import StudentLab, RobotLab
StudentLab('/inputs/model','/inputs/parent') loads the ordinary-practice executive.
The original and taught adapters are mounted separately; choose your starting point.
RobotLab('/inputs/motor','/work/robot').reset() creates a development mission.
You have unrestricted access to the returned lab's underlying simulation and model
objects. Create your own scenes or training code. No demonstration curriculum is
mandatory. The starter /work/candidate/policy.py illustrates the deployment API.
For final deployment the only supplied dependencies are /inputs/code,
/inputs/model and /inputs/parent. Package any other weights, hooks or data in
your candidate directory. Policy.reset gets public mission information;
Policy.act gets {'observation': ..., 'documentation': ..., 'finished': ...}.
Return one action dict or its JSON string. Do not print diagnostics to stdout
inside Policy: stdout is the action protocol. Use stderr for diagnostics.
Do not assume a frozen KV prefix constitutes durable weight learning. Describe
all added policies. Qwen must make the adaptive decisions for the primary claim.
'''
    config=dict(resources=resources,readonly=readonly,runtime=linux['runtime'],python_home=linux['python_home'],
        linux_runtime_manifest=str(linux_manifest),linux_runtime_sha256=digest(linux_manifest),
        runtime_verifier=str(ROOT/'tools/verify_supervisor_runtime.py'),
        cli=old['cli'],catalog=str(catalog),instructions=str(instructions),brief=str(brief),
        template=str(ROOT/'tools/supervisor_candidate_template.py'),job_worker=str(ROOT/'tools/supervisor_job_worker.py'),
        final_worker=str(ROOT/'tools/supervisor_final_worker.py'),deployment_worker=str(ROOT/'tools/supervisor_deployment_worker.py'),
        workbench_help=help_text,limits=dict(wall_seconds=72000,gpu_seconds=28800,calls=64))
    preview=data/'prompt_preflight';preview.mkdir();(preview/'empty').mkdir()
    preflight(config['cli'],catalog,instructions,preview,brief.read_text()+help_text+TOOL_INSTRUCTIONS)
    spec=dict(model_id='astra-supervisor-v2',source_roots=['src','experiments','tools','tests'],
        constructor=dict(module='skillforge.supervisor_session',symbol='SupervisorStudy',source='src/skillforge/supervisor_session.py'),
        constructor_kwargs={},runner=dict(module='run_astra_supervisor',symbol='run',source='experiments/run_astra_supervisor.py'),runner_config=config)
    save(ROOT/'experiments/source_pins/astra_supervisor_v2.spec.json',spec)
    report=dict(status='prepared',approved=False,resources=len(resources),linux_resources=len(linux['files']),model_calls=0,
                author_transport='full visible transcript plus isolated general Python tool; private reasoning not resumed')
    save(ROOT/'docs/wspl/supervisor_preparation.json',report);print(json.dumps(report))


if __name__=='__main__':main()
