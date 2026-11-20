"""Explicit recovery of a failed collection; never replay completed tool jobs."""
import json
from pathlib import Path
import shutil
import time
from .supervisor_workspace import inventory,save


def restore(parent,out,brief,limits):
    parent=Path(parent);out=Path(out)
    state=json.loads((parent/'supervisor/session.json').read_text())
    budget=json.loads((parent/'budget.json').read_text())
    status=json.loads((parent/'status.json').read_text())
    gate=json.loads((parent/'persistence_check/receipt.json').read_text())
    if state['phase']!='interrupted' or state['submission'] is not None:
        raise ValueError('Recovery requires an interrupted, unsubmitted session')
    if state['brief']!=brief or budget['limits']!=limits or budget['active'] is not None:
        raise ValueError('Changed brief/budget or unresolved active job')
    if status['status']!='failed' or gate['status']!='passed':
        raise ValueError('Parent must have passed persistence and stopped')
    turn=parent/'supervisor'/f"turn_{state['turn']:03d}"
    if (turn/'response.txt').exists() or (turn/'executed.py').exists():
        raise ValueError('Interrupted turn already produced a response or execution')
    events=[json.loads(x) for x in (turn/'collection/events.jsonl').read_text().splitlines() if x.strip()]
    if not any(e.get('type')=='error' and 'at capacity' in e.get('message','') for e in events):
        raise ValueError('Expected archived model-capacity failure')
    if any(e.get('type')=='turn.completed' or e.get('item',{}).get('type')=='agent_message' for e in events):
        raise ValueError('Ambiguous completed response in interrupted collection')
    if state['transcript'][-1]['role']!='tool':raise ValueError('Not at a completed tool boundary')
    # This boundary has no images; never silently rewrite transcript references.
    if any(isinstance(x.get('content'),dict) and x['content'].get('archived_images') for x in state['transcript']):
        raise ValueError('Image-bearing recovery needs explicit image bindings')
    prompt=state['brief']+'\n'
    from .supervisor_session import TOOL_INSTRUCTIONS
    prompt+=TOOL_INSTRUCTIONS+'\nComplete visible history:\n'+json.dumps(state['transcript'])
    if (turn/'collection/prompt.txt').read_text(encoding='utf8')!=prompt:
        raise ValueError('Saved next prompt differs')
    before=inventory(parent/'development')
    shutil.copytree(parent/'development',out/'development')
    if inventory(out/'development')!=before:raise ValueError('Workspace copy differs')
    elapsed=float(status['elapsed_seconds'])
    if not 0<=elapsed<limits['wall_seconds']:raise ValueError('Parent wall budget exhausted')
    # Downtime excluded; all recorded elapsed time, calls (including failed), and
    # GPU leases remain charged. No fresh scientific budget.
    budget['started']=time.time()-elapsed
    save(out/'budget.json',budget)
    state['phase']='ready';state.pop('error',None)
    (out/'supervisor').mkdir();save(out/'supervisor/session.json',state)
    shutil.copytree(parent/'persistence_check',out/'persistence_check')
    save(out/'recovery_receipt.json',dict(parent=str(parent.resolve()),
        completed_research_responses=state['turn'],calls_already_used=budget['calls_used'],
        maximum_additional_calls=limits['calls']-budget['calls_used'],
        previous_elapsed_seconds=elapsed,workspace_files=len(before),
        prompt_identical=True,persistence_reused=True))
