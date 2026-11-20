"""Host Linux worker: config supplied by the supervisor, never by the candidate."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.supervisor_workspace import Budget,sandbox_command,run_job


def main():
    c=json.loads(Path(sys.argv[1]).read_text())
    budget=Budget(c['ledger'],**c['limits'])
    argv=sandbox_command(c['workspace'],c['runtime'],c['command'],c.get('readonly'),
                         c['gpu'],c.get('python_home'))
    result=run_job(budget,c['label'],argv,c['logs'],c['gpu'],c['seconds'])
    print(json.dumps(result))


if __name__=='__main__':main()
