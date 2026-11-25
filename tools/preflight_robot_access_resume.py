"""Offline recovery prompt and file/sandbox fixtures; no hosted requests/forwards."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.robot_access_recovery import restore_author
from skillforge.robot_authoring_access_resume import prompt
from skillforge.robot_collector import preflight
from skillforge.author_python_workspace import PythonWorkspace
from skillforge.reliable_json import save


def main():
    old=ROOT.parent/'WSPL-robot-authoring-access-v1';oldout=old/'results/robot_authoring_access_v1'
    inv=json.loads((ROOT/'docs/wspl/robot_access_resume_v1_inventory.json').read_text())
    data=ROOT/'data/robot_access_resume_v1';data.mkdir(parents=True,exist_ok=True)
    for name in ('catalog.json','instructions.md'):shutil.copyfile(old/'data/robot_authoring_access_v1'/name,data/name)
    cfg=json.loads((old/'experiments/source_pins/robot_authoring_access_v2.manifest.json').read_text())['runner_config']
    folder=Path(tempfile.mkdtemp(prefix='recovery_',dir=data))
    # Copy actual historical inputs/work without executing any author construction.
    workspace=SimpleNamespace(root=folder/'copy',inputs=folder/'copy/inputs',work=folder/'copy/work',profile=SimpleNamespace(grant=lambda *a,**k:None))
    workspace.inputs.mkdir(parents=True);workspace.work.mkdir()
    restored=restore_author(workspace,inv['resume'])
    catalog=json.loads((oldout/'common_inputs/catalog.json').read_text())
    packet=dict(adapter_id=catalog['adapter_id'],projection_count=len(catalog['projections']),element_count=catalog['element_count'],
                projections=list(catalog['projections']),acquisition_rows=255,parent_development=json.loads((oldout/'common_inputs/parent_development.json').read_text()))
    prompt_folder=folder/'prompt';prompt_folder.mkdir();(prompt_folder/'empty').mkdir()
    preflight(cfg['cli'],data/'catalog.json',data/'instructions.md',prompt_folder,
              prompt('native',workspace.inputs,packet,restored['transcript'])+'\n'+restored['notice'],profile='supervisor')
    # Physical ACL check uses synthetic historical data, never attempts to alter real history.
    fake=folder/'synthetic_old';(fake/'sandbox/inputs').mkdir(parents=True);(fake/'sandbox/work').mkdir()
    (fake/'sandbox/inputs/data.txt').write_text('synthetic input')
    (fake/'sandbox/work/keep.txt').write_text('synthetic history')
    tr=[]
    for _ in range(21):tr.extend([dict(role='assistant',content='{}'),dict(role='tool',content={})])
    save(fake/'transcript.json',tr);save(fake/'status.json',dict(turns=21,proposal_attempts=2,inspection_attempts=5,candidate=None))
    real=PythonWorkspace(folder/'sandbox',cfg['native_runtime'])
    restore_author(real,dict(author_folder=str(fake),consumed_responses=22))
    code='from pathlib import Path\np=Path('+repr(str(fake/'sandbox/work/keep.txt'))+')\nassert p.read_text()=="synthetic history"\nassert Path("keep.txt").read_text()=="synthetic history"\nPath("keep.txt").write_text("new copy")\ntry:\n p.write_text("bad")\n raise AssertionError("historical path writable")\nexcept PermissionError: print("OLD_READONLY_NEW_WRITABLE")\n'
    check=real.execute(code,timeout=60)
    assert check['exit_code']==0 and not check['timed_out']
    assert (fake/'sandbox/work/keep.txt').read_text()=='synthetic history'
    assert (real.work/'keep.txt').read_text()=='new copy'
    report=dict(status='passed',author_calls=0,scientific_predictions=0,actual_history_entries=len(restored['transcript']),
                consumed_responses=restored['start_turn'],remaining_responses=24-restored['start_turn'],
                actual_files_copied_verified=True,actual_offline_prompt=True,old_absolute_paths_readonly=True,
                new_workspace_writable=True,prior_queries=restored['queries'],prior_evaluations=restored['attempts'],folder=str(folder))
    save(ROOT/'docs/wspl/robot_access_resume_v1_preflight.json',report);print(json.dumps(report))


if __name__=='__main__':main()
