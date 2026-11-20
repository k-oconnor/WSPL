"""Hash Linux runtime and its system libraries without running an experiment."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from skillforge.supervisor_workspace import digest,save

roots=[Path(sys.prefix),Path(sys.base_prefix),Path('/usr/lib/x86_64-linux-gnu'),Path('/usr/lib/wsl')]
files={Path('/usr/bin/bwrap')}
for root in roots:
    files.update(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
records={str(p):digest(p) for p in sorted(files)}
save(Path(sys.argv[1]),dict(runtime=sys.prefix,python_home=sys.base_prefix,files=records))
print('Bound Linux files:',len(records))
