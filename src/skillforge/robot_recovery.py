"""Reuse only hash-verified completed work from the interrupted first study."""
import json
from pathlib import Path
import shutil
from .fresh_robot import sha
from .robot_submissions import load_arrays
from .wspl_lora import identity
from .reliable_json import save


def load_prefix(config, out, guard):
    recovery=json.loads(Path(config['recovery_manifest']).read_text())
    prior=Path(recovery['prior_root']).resolve(strict=True)
    for path,expected in recovery['files'].items():
        guard()
        resolved=Path(path).resolve(strict=True)
        if not resolved.is_relative_to(prior) or sha(resolved)!=expected:
            raise ValueError('Recovery artifact changed or outside prior output')
    common=Path(out)/'common_inputs';common.mkdir()
    for name,expected in recovery['common_input_hashes'].items():
        if Path(name).name!=name:raise ValueError('Invalid common input filename')
        shutil.copyfile(prior/'common_inputs'/name,common/name)
        if sha(common/name)!=expected:raise ValueError('Copied input identity mismatch')
    save(Path(out)/'common_input_hashes.json',recovery['common_input_hashes'])
    parent=load_arrays(common/'parent.npz')
    gradient=prior/'gradient/candidate.npz'
    if identity(parent)!=recovery['parent_adapter_id'] or identity(load_arrays(gradient))!=recovery['gradient_adapter_id']:
        raise ValueError('Reused factor identity mismatch')
    rows=json.loads((common/'examples.json').read_text())
    catalog=json.loads((common/'catalog.json').read_text())
    base_dev=json.loads((prior/'parent_development/episodes.json').read_text())
    if len(rows)!=255 or len(base_dev)!=16 or catalog['adapter_id']!=recovery['parent_adapter_id']:
        raise ValueError('Incomplete recovery prefix')
    save(Path(out)/'reused_prefix.json',dict(prior_root=str(prior),prior_manifest_sha256=recovery['prior_manifest_sha256'],
        gradient=str(gradient),gradient_adapter_id=recovery['gradient_adapter_id'],
        acquisition_decisions=len(rows),development_missions=len(base_dev),new_training_updates=0,
        prior_author_invocations=1,prior_completed_author_responses=0,prior_author_usage='unknown'))
    return common,rows,catalog,base_dev,gradient
