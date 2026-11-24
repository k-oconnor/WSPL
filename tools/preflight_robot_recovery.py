"""Verify and copy the completed real prefix, without model construction."""
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.robot_recovery import load_prefix
from skillforge.reliable_json import save
from skillforge.fresh_robot import sha

root=Path(tempfile.mkdtemp(prefix='recovery_fixture_',dir=ROOT/'data/robot_authoring_v1'))
path=ROOT/'docs/wspl/robot_authoring_v2_recovery_inputs.json'
common,rows,catalog,development,gradient=load_prefix({'recovery_manifest':str(path)},root,lambda:None)
report=dict(status='passed',folder=str(root),acquisition_decisions=len(rows),development_missions=len(development),
            gradient_file_sha256=sha(gradient),parent_adapter_id=catalog['adapter_id'],
            model_constructors=0,new_training_updates=0,new_predictions=0,author_calls=0)
save(ROOT/'docs/wspl/robot_authoring_v2_recovery_preflight.json',report)
print(json.dumps(report))
