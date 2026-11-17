"""Prepare an unapproved controller-author specification and resource identities."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path('C:/Users/kevin/skillforge-runtime/models/Qwen2.5-Coder-1.5B-Instruct')
OLD_SPEC = Path('C:/Users/kevin/OneDrive/Desktop/wspl_projects/Skillforge-authoring-sft-v2/experiments/source_pins/authoring_sft_wspl_v2.spec.json')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    previous = json.loads(OLD_SPEC.read_text())['runner_config']['resources']
    paths = {Path(path) for path in previous if ('venv-smollm' in path or 'PythonSoftwareFoundation.Python' in path) and Path(path).is_file()}
    paths.update(path for path in MODEL.glob('*') if path.is_file() and path.suffix in ('.json', '.safetensors', '.txt'))
    paths.add(Path(sys.executable))
    paths.add(ROOT / 'docs/wspl/CONTROLLER_AUTHOR_V1.md')
    resources = {str(path.resolve()): digest(path) for path in sorted(paths)}
    spec = dict(model_id='controller-author-v1', source_roots=['src', 'experiments', 'tools', 'tests'],
        constructor=dict(module='skillforge.authoring_sft', source='src/skillforge/authoring_sft.py', symbol='AuthoringSFTStudy'),
        constructor_kwargs=dict(model_path=str(MODEL), seed=113, rank=16, alpha=32),
        runner=dict(module='run_controller_author', source='experiments/run_controller_author.py', symbol='run'),
        runner_config=dict(context=2048, accumulation=8, epochs=2, lr=.0002, warmup=20,
                           order_seed=711, wall_guard_seconds=21600, resources=resources))
    path = ROOT / 'experiments/source_pins/controller_author_v1.spec.json'
    path.write_bytes((json.dumps(spec, indent=2, sort_keys=True) + '\n').encode())
    print(json.dumps(dict(spec=str(path), resources=len(resources), approved=False)))


if __name__ == '__main__':
    main()
