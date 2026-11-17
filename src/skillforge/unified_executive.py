"""Matched executive repeat using the unified compiler and fixed module ABI."""
import re
import json
from pathlib import Path

from . import executive_study as legacy
from .wspl_language import compile_source, library_document, digest


def header(library):
    return 'wspl 0.3;\nuse robot = library("planar", sha256="'+digest(library)+'");\n'


def teacher(task, library):
    lines = [header(library)+'task Mission {']
    lines += ['  robot.transfer('+n+', '+task['goals'][n]+');' for n in legacy.order_goals(task)]
    return '\n'.join(lines+['  robot.move_to(home);', '}'])+'\n'


def prompt(task, library):
    visible = {k:v for k,v in task.items() if k not in ('jammed','panel')}
    return ('Write only one WSPL 0.3 task program. Begin with this exact header:\n'+header(library)+
        'Then write task Mission { ... }. Statements end in semicolons. Allowed calls: '
        'robot.move_to(OBJECT_OR_POINT); robot.grasp(OBJECT); robot.release(OBJECT); '
        'robot.transfer(OBJECT, POINT);. These call a pinned library of six typed neural modules. '
        'transfer expands into approach, grasp, transport and release. Each skill completes before the next. '
        'Move every named object to its assigned goal and leave it there, obey every before pair [A,B] '
        '(A must be delivered before B), and return the empty gripper to home. '
        'Among currently available goals choose the alphabetically first object. '
        'Do not move objects again after delivery. All positions are normalized planar coordinates. '
        'The frozen motor skills handle trajectories. No explanations or code fences.\nScene and task: '+legacy.canonical(visible))


def dataset(library):
    data = legacy.dataset()
    for rows in data.values():
        for row in rows:
            row['messages'] = [dict(role='user',content=prompt(row['task'],library)),
                               dict(role='assistant',content=teacher(row['task'],library))]
    return data


def reused_replies(config, test_rows):
    historical=[json.loads(line) for line in Path(config['previous_replies']).read_text(encoding='utf8').splitlines()]
    historical=[r for r in historical if r['arm']=='trained']
    expected={r['id']:r['task'] for r in test_rows}
    if len(historical)!=len(expected) or {r['id'] for r in historical}!=set(expected):
        raise ValueError('Historical reply task identities differ')
    previous_tasks=[json.loads(line) for line in Path(config['previous_tasks']).read_text(encoding='utf8').splitlines()]
    if len(previous_tasks)!=len(expected) or {r['id']:r['task'] for r in previous_tasks}!=expected:
        raise ValueError('Historical task content differs')
    return [dict(r,arm='previous',reused=True) for r in historical]


def score(source, task, blocks, specs, keep_trace=True):
    text = source.strip()
    fence = re.fullmatch(r'```(?:wspl)?\s*\n([\s\S]*?)\n```',text)
    if fence: text=fence.group(1)
    library = library_document(blocks,specs)
    try:
        compiled = compile_source(text, libraries={'planar':library}, objects=task['objects'],
                                  points=task['points'],limit=64)
        if compiled.task_name != 'Mission' or compiled.modules or compiled.edit_plan:
            raise ValueError('Executive may compose the frozen library only')
        if compiled.imports != {'robot':dict(key='planar',sha256=digest(library))}:
            raise ValueError('Executive must bind the declared robot library')
    except (ValueError,TypeError,KeyError) as error:
        return dict(constructed=False,task_success=False,safe_failure=False,panel=task['panel'],error=str(error))
    result = legacy.score('',task,blocks,specs,keep_trace,compiled_program=compiled.calls)
    result['compilation'] = compiled.provenance
    return result
