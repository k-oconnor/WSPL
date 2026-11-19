"""Synthetic mission/scorer checks, no teacher/student model calls or reserved seeds."""
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.executive_teaching import TeachingMission,CONDITIONS,ordinary_action
from skillforge.reliable_json import save


def main():
    artifact=ROOT.parent/'WSPL-factory-v1/results/factory_pilot_v1/artifact_final.json'
    root=ROOT/'data/teaching_preflight';root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='physics_',dir=root));checks=[]
    for i,c in enumerate(CONDITIONS):
        mission=TeachingMission(artifact,folder/c,991+i,3,c,32)
        try:
            assert mission.session.steps==0 and mission.episode.selected is None
            while not mission.episode.finished:
                # Explicit engineering oracle is never supplied to teacher or student.
                needed=mission.compensated if ((c=='changed') or (c=='switch' and mission.changed) or (c=='return' and not mission.changed)) else mission.nominal
                public=mission.public();o=public['observation']
                if o['selected_module']!=needed:a=dict(op='select',module=needed)
                else:
                    pending=[g for g in o['requested_order'] if g not in o['observed_visits']]
                    a=dict(op='reach',goal=pending[0]) if pending else dict(op='stop')
                before=mission.session.steps;mission.execute(json.dumps(a))
                if a['op']=='select':assert mission.session.steps==before
            result=mission.result();assert result['success'],(c,result)
            assert mission.changed==(c in ('switch','return'))
            checks.append(dict(condition=c,success=result['success'],decisions=result['decisions'],ticks=result['physical_ticks']))
        finally:mission.close()
    report=dict(status='passed',engineering_only=True,conditions=checks,model_calls=0)
    save(ROOT/'docs/wspl/teaching_physics_preflight.json',report);print(json.dumps(report))


if __name__=='__main__':main()
