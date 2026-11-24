"""Independent identical Python/wheel copies, physically omitting the WSPL package."""
import json
from pathlib import Path
import shutil

source=Path(r'C:\Users\kevin\wspl-author-python-v1\runtime')
target=Path(r'C:\Users\kevin\wspl-author-native-v1\runtime')
if target.exists():raise ValueError('Fresh destination required')
def ignore(folder,names):
    return ['skillforge'] if Path(folder)==source/'Lib/site-packages' else []
shutil.copytree(source,target,ignore=ignore)
report=dict(purpose='native comparator: identical runtime minus WSPL package',source=str(source),
            destination=str(target),hardlinks=False,omitted='Lib/site-packages/skillforge')
(target/'native_build_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
