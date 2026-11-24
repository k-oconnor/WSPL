"""One generic transport healthcheck; no research data, edits, or task scoring."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.robot_collector import collect
from skillforge.reliable_json import save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--live-healthcheck',action='store_true');a=parser.parse_args()
    if not a.live_healthcheck:raise ValueError('Explicit healthcheck flag required; no study is launched')
    data=ROOT/'data/robot_authoring_v1'
    folder=Path(tempfile.mkdtemp(prefix='transport_health_',dir=data))/'collection'
    answer,receipt=collect(r'C:\Users\kevin\.vscode\extensions\openai.chatgpt-26.1117.41037-win32-x64\bin\windows-x86_64\codex.exe',
        data/'catalog.json',data/'instructions.md',folder,
        'Return exactly the text WSPL_TRANSPORT_OK. Do not use tools.',120,lambda:False,profile='supervisor')
    if answer.strip()!='WSPL_TRANSPORT_OK':raise ValueError('Healthcheck response mismatch')
    result=dict(status='passed',purpose='generic collector transport healthcheck',scientific_author_requests=0,
                research_data_supplied=False,task_predictions=0,folder=str(folder),answer=answer,receipt=receipt)
    save(ROOT/'docs/wspl/robot_collector_transport_v2_healthcheck.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
