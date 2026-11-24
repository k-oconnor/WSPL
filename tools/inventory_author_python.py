"""Record the exact built runtime; this is not a scientific source approval."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
runtime = root/'runtime'
def entry(path):
    if path.is_symlink() or path.stat().st_nlink != 1:
        raise ValueError('Independent regular files required: '+str(path))
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(runtime).as_posix(), size=path.stat().st_size, sha256=sha)
with ThreadPoolExecutor(max_workers=8) as pool:
    files = list(pool.map(entry, sorted(p for p in runtime.rglob('*') if p.is_file())))
manifest = dict(format='wspl.author-python-runtime.v1', root=str(runtime), files=files,
                scientific_approval=False)
encoded = (json.dumps(manifest, sort_keys=True, separators=(',', ':'))+'\n').encode()
(root/'environment_manifest.json').write_bytes(encoded)
print(json.dumps(dict(files=len(files), bytes=sum(v['size'] for v in files),
                      manifest_sha256=hashlib.sha256(encoded).hexdigest(),
                      path=str(root/'environment_manifest.json'))))
