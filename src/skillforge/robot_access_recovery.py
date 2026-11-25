"""Restore an interrupted author's files and visible history without extra budget."""
import json
from pathlib import Path
import shutil
from .fresh_robot import sha
from .reliable_json import save


def copy_regular_tree(source,target):
    source=Path(source).resolve(strict=True);target=Path(target)
    for path in sorted(source.rglob('*')):
        if path.is_symlink() or not path.resolve().is_relative_to(source):raise ValueError('Linked recovery path')
        relative=path.relative_to(source);dest=target/relative
        if path.is_dir():dest.mkdir(exist_ok=True)
        elif path.is_file():
            if path.stat().st_nlink!=1:raise ValueError('Hard-linked recovery file')
            shutil.copyfile(path,dest)
            if sha(path)!=sha(dest):raise ValueError('Recovery copy mismatch')
        else:raise ValueError('Unknown recovery entry')


def restore_author(workspace,resume):
    prior=Path(resume['author_folder']).resolve(strict=True)
    transcript=json.loads((prior/'transcript.json').read_text())
    local=json.loads((prior/'status.json').read_text())
    used=resume['consumed_responses']
    if used!=local['turns']+1 or len(transcript)!=2*local['turns'] or used>=24:
        raise ValueError('Recovery response accounting mismatch')
    if any(x['role']!=('assistant' if i%2==0 else 'tool') for i,x in enumerate(transcript)):
        raise ValueError('Incomplete visible history')
    for name in ('inputs','work'):
        copy_regular_tree(prior/'sandbox'/name,getattr(workspace,name))
        # Historical absolute paths remain readable, never writable by the new profile.
        workspace.profile.grant(prior/'sandbox'/name)
    note=(f'Resuming this SAME author after a usage-limit interruption. {local["turns"]} responses completed; '
          f'the next request failed before an answer and consumed response slot{used}. '
          f'Only {24-used} responses remain. Prior queries/evaluations remain charged. '
          'Your complete work and input files have been copied into this new sandbox. '
          'Use the current working directory for writes and relative submission paths. '
          'Prior absolute input/work paths remain read-only for compatibility. '
          'No other author results or final data are provided.')
    save(workspace.root/'recovery.json',dict(**resume,prior_status=local,notice=note))
    return dict(transcript=transcript,accepted=local['candidate'],attempts=local['proposal_attempts'],
                queries=local['inspection_attempts'],start_turn=used,notice=note)
