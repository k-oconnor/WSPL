"""Fingerprint or verify the separate CPU Python-construction runtime on Linux."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--write');p.add_argument('--verify');args=p.parse_args()
    if bool(args.write)==bool(args.verify):raise ValueError('choose one operation')
    if args.write:
        roots=[Path('/home/kevin/venvs/skillforge-author'),Path('/usr/lib/python3.12')]
        paths={Path('/usr/bin/bwrap'),Path('/usr/bin/python3'),Path('/usr/bin/python3.12'),Path('/home/kevin/venvs/skillforge-author/bin/python')}
        for root in roots:
            paths.update(p for p in root.rglob('*') if p.is_file() and (p.suffix in ('.py','.so','.cfg') or '.so.' in p.name))
        for name in ('libc.so.6','libm.so.6','libz.so.1','libexpat.so.1','libffi.so.8','libpthread.so.0','libdl.so.2','librt.so.1','libgcc_s.so.1','ld-linux-x86-64.so.2'):
            paths.add(Path('/lib/x86_64-linux-gnu')/name)
        records={str(p):sha(p) for p in sorted(paths)}
        with Path(args.write).open('x') as f:json.dump(records,f,indent=2,sort_keys=True)
        print('Bound',len(records),'Linux runtime files')
    else:
        records=json.loads(Path(args.verify).read_text())
        for p,digest in records.items():
            if sha(p)!=digest:raise ValueError('runtime file differs: '+p)
        print('Verified',len(records),'Linux runtime files')


if __name__=='__main__':main()
