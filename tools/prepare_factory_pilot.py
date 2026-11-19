"""Bind actual local runtime and offline text/image preflight; no model calls."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from skillforge.codex_one_shot import preflight
from skillforge.factory_pilot import packet
from skillforge.reliable_json import save


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    old=ROOT.parent/'WSPL-astra-retention-v1/results/astra_retention_v1/approved_manifest.json'
    assert sha(old)=='4ca2229cd1bcdd9852accad248b34eb34edf335f1436d6b2a98aa9d109874eb4'
    c=json.loads(old.read_text())['runner_config']
    data=ROOT/'data/factory_v1';data.mkdir(parents=True,exist_ok=True)
    catalog=data/'catalog_frozen.json';catalog.write_bytes(Path(c['catalog']).read_bytes())
    instructions=data/'instructions.md';instructions.write_text("Answer the user's request.\n",encoding='utf8')
    preview=data/('preflight_'+str(time.time_ns()));preview.mkdir();(preview/'empty').mkdir()
    fixtures=sorted((ROOT/'data/factory_preflight').glob('*/report.json'),key=lambda p:p.stat().st_mtime)
    report=json.loads(fixtures[-1].read_text());assert len(report['synthetic_controller_fixtures'])==3
    frame=Path(report['folder'])/'observations/frame_0000.png'
    preflight(c['cli'],catalog,instructions,preview,packet([],dict(responses=12,evaluations=2)),[frame])
    resources={}
    # Full installed runtime, native libraries and simulator assets; no model/data bundles.
    roots=[Path(sys.prefix)/'Lib/site-packages',Path(sys.base_prefix)/'DLLs',Path(sys.base_prefix)/'Lib']
    for root in roots:
        for p in sorted(root.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):
                resources[str(p.resolve())]=sha(p)
    for p in [Path(sys.executable),Path(sys.base_prefix)/'python311.dll',Path(c['cli']),catalog,instructions,
              ROOT/'docs/wspl/FACTORY_PILOT_V1.md']:
        if not p.is_file():raise ValueError('Required resource missing '+str(p))
        resources[str(p.resolve())]=sha(p)
    config=dict(resources=resources,cli=c['cli'],catalog=str(catalog),instructions=str(instructions),
        changed_mapping=[[0,-.8,0],[1.2,0,0],[0,0,.9]],
        offsets=[[.04,0,.02],[-.04,0,.02],[0,.04,.02],[0,-.04,.02],
                 [.03,.03,.02],[-.03,.03,.02],[.03,-.03,.02],[-.03,-.03,.02]],wall_guard_seconds=14400)
    spec=dict(model_id='factory-pilot-v1',source_roots=['src','experiments','tools','tests'],
        constructor=dict(module='skillforge.factory_pilot',symbol='FactoryStudy',source='src/skillforge/factory_pilot.py'),
        constructor_kwargs={},runner=dict(module='run_factory_pilot',symbol='run',source='experiments/run_factory_pilot.py'),runner_config=config)
    save(ROOT/'experiments/source_pins/factory_pilot_v1.spec.json',spec)
    result=dict(status='prepared',approved=False,resources=len(resources),
                offline_image_bytes_verified=True,physics_preflight=report,author_calls=0)
    save(ROOT/'docs/wspl/factory_pilot_preparation.json',result)
    print(json.dumps(dict(resources=len(resources),offline_image_bytes_verified=True,author_calls=0)))


if __name__=='__main__':main()
