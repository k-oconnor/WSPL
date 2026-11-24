"""Copy only the WSPL LoRA bridge's local import closure into a dedicated runtime."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys


def main():
    source = Path(__file__).resolve().parents[1]/'src/skillforge'
    target = Path(sys.argv[1])/'Lib/site-packages/skillforge'
    target.mkdir(parents=True, exist_ok=True)
    pending, done = ['__init__', 'wspl_lora'], {}
    while pending:
        name = pending.pop()
        if name in done:
            continue
        path = source/(name+'.py')
        code = path.read_bytes()
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1 or not node.module or '.' in node.module:
                    raise ValueError(f'Unsupported local import in {name}')
                pending.append(node.module)
        shutil.copy2(path, target/path.name)
        done[name] = hashlib.sha256(code).hexdigest()
    report = dict(source=str(source), target=str(target), files=done)
    (target.parent.parent.parent/'wspl_package_receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
