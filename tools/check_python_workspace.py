"""Two synthetic full-Python calls, including filesystem persistence and errors."""
import json
from pathlib import Path
import sys
import uuid
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from skillforge.author_python_workspace import PythonWorkspace

root = Path(r'C:\Users\kevin\wspl-author-python-v1')
session = PythonWorkspace(root/('session-check-'+uuid.uuid4().hex[:10]), root/'runtime')
first = session.execute("import numpy as np\nnp.save('persist.npy', np.arange(6))\nprint('created array')\n")
second = session.execute("import numpy as np\nx=np.load('persist.npy', allow_pickle=False)\nassert x.sum()==15\nprint('persistent sum:', x.sum())\n")
error = session.execute("raise ValueError('synthetic error test')\n")
checks = dict(first=first['exit_code']==0 and 'created array' in first['stdout']['text'],
              second=second['exit_code']==0 and 'persistent sum: 15' in second['stdout']['text'],
              error=error['exit_code']!=0 and 'synthetic error test' in error['stdout']['text'])
result = dict(root=str(session.root), checks=checks, passed=all(checks.values()))
(session.root/'engineering_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
if not result['passed']:
    raise SystemExit(1)
