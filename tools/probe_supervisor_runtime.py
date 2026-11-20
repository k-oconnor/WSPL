"""Read-only dependency/device inventory; no model load or scientific evaluation."""
import importlib.util
import json
import sys
from pathlib import Path

names=('torch','transformers','mujoco','robosuite','numpy','PIL')
result=dict(executable=sys.executable,prefix=sys.prefix,base_prefix=sys.base_prefix,
            packages={n:bool(importlib.util.find_spec(n)) for n in names})
if result['packages']['torch']:
    import torch
    result.update(torch=torch.__version__,cuda=torch.cuda.is_available())
print(json.dumps(result))
