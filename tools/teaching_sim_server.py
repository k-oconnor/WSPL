"""Private physics worker; called by the approved runner. JSON-line RPC, no model."""
import argparse
import gzip
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from source_consistency import verify,CheckedImports


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--sha256',required=True)
    a=parser.parse_args();manifest=verify(ROOT,Path(a.manifest),a.sha256)
    sys.meta_path.insert(0,CheckedImports(ROOT,manifest))
    from skillforge.executive_teaching import TeachingMission
    mission=None
    try:
        for line in sys.stdin:
            try:
                r=json.loads(line)
                if r['op']=='start':
                    if mission is not None:mission.close()
                    mission=TeachingMission(r['artifact'],r['output'],r['seed'],r['count'],r['condition'],r['max_decisions'])
                    result=mission.public()
                elif r['op']=='act':result=mission.execute(r['action'])
                elif r['op']=='finish':
                    result=mission.result()
                    private={k:result.pop(k) for k in ('trace','decision_trace','sensor_trace')}
                    path=Path(r['trace_path']);path.write_bytes(gzip.compress(json.dumps(private).encode(),mtime=0))
                    result['trace_path']=str(path);mission.close();mission=None
                elif r['op']=='close':
                    print(json.dumps(dict(ok=True,result={})),flush=True);break
                else:raise ValueError('Unknown RPC')
                print(json.dumps(dict(ok=True,result=result)),flush=True)
            except Exception as error:
                print(json.dumps(dict(ok=False,error=repr(error))),flush=True)
                break
    finally:
        if mission is not None:mission.close()


if __name__=='__main__':main()
