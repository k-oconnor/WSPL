"""Real isolated physics-worker transport, engineering fixture only."""
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import inventory,write_json,digest
from run_executive_teaching import Simulator
from skillforge.reliable_json import save


def main():
    root=ROOT/'data/teaching_preflight';root.mkdir(parents=True,exist_ok=True)
    folder=Path(tempfile.mkdtemp(prefix='rpc_',dir=root))
    manifest=folder/'engineering_manifest.json'
    scopes=['src','experiments','tools','tests']
    write_json(manifest,dict(schema=1,model_id='engineering-only',source_roots=scopes,files=inventory(ROOT,scopes),
        constructor=dict(module='skillforge.authoring_sft',symbol='AuthoringSFTStudy',source='src/skillforge/authoring_sft.py'),
        runner=dict(module='run_executive_teaching',symbol='run',source='experiments/run_executive_teaching.py')))
    config=dict(sim_python='C:/Users/kevin/skillforge-runtime/venv-sim/Scripts/python.exe',sim_worker=str(ROOT/'tools/teaching_sim_server.py'))
    sim=Simulator(config,folder,manifest,digest(manifest))
    try:
        public=sim.call(dict(op='start',artifact=str(ROOT.parent/'WSPL-factory-v1/results/factory_pilot_v1/artifact_final.json'),output=str(folder/'scene'),seed=995,count=5,condition='ordinary',max_decisions=8))
        assert public['observation']['telemetry']['sequence']==0 and public['observation']['selected_module'] is None
        assert set(public)=={'observation','documentation','finished'}
        module=public['observation']['available_modules'][0]
        selected=sim.call(dict(op='act',action=json.dumps(dict(op='select',module=module))))
        assert selected['observation']['telemetry']['sequence']==0
        moved=sim.call(dict(op='act',action=json.dumps(dict(op='probe',command=[.02,0,0],ticks=2))))
        assert moved['observation']['telemetry']['sequence']==2 and moved['observation']['selected_module']==module
        save(ROOT/'data/teaching_preflight/public_fixture.json',public)
        report=dict(status='passed',engineering_only=True,source_checked_child=True,private_fields_absent=True,
                    no_startup_action=True,explicit_probe_ticks=2)
        save(ROOT/'docs/wspl/teaching_rpc_preflight.json',report);print(json.dumps(report))
    finally:sim.close()


if __name__=='__main__':main()
