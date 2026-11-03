"""Fail-closed experiment launcher. Approval pins are separate from source snapshots."""
import argparse
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import inspect
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / 'experiments/source_pins/approved.json'

class ConsistencyError(RuntimeError):
    pass

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    # Stable source-manifest bytes across Windows/Linux checkouts.
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8'))

def contained(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or '..' in Path(relative).parts or not path.is_relative_to(root.resolve()):
        raise ConsistencyError(f'Source path escapes repository: {relative}')
    return path

def inventory(root, scopes):
    result = {}
    for scope in scopes:
        folder = contained(root, scope)
        if not folder.is_dir():
            raise ConsistencyError(f'Missing source scope: {scope}')
        for path in sorted(folder.rglob('*.py')):
            if '__pycache__' in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            contained(root, relative)  # Reject symlinks outside the repository.
            result[relative] = digest(path)
    return result

def verify(root, manifest_path, expected_digest):
    if digest(manifest_path) != expected_digest:
        raise ConsistencyError('Manifest does not match the independently approved SHA-256 pin')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 1:
        raise ConsistencyError('Unsupported source-pin schema')
    actual = inventory(root, manifest['source_roots'])
    if actual != manifest['files']:
        changed = sorted(k for k in set(actual) | set(manifest['files']) if actual.get(k) != manifest['files'].get(k))
        raise ConsistencyError('Source inventory mismatch: ' + ', '.join(changed[:12]))
    for role in ['constructor', 'runner']:
        binding = manifest[role]
        if binding['source'] not in manifest['files']:
            raise ConsistencyError(f'{role} source is not pinned')
    return manifest

class CheckedLoader(importlib.machinery.SourceFileLoader):
    """Compile verified source bytes; never execute stale .pyc bytecode."""
    def __init__(self, name, path, expected):
        super().__init__(name, path)
        self.expected = expected

    def get_code(self, fullname):
        source = Path(self.path).read_bytes()
        if hashlib.sha256(source).hexdigest() != self.expected:
            raise ConsistencyError(f'Source changed during import: {self.path}')
        return compile(source, self.path, 'exec', dont_inherit=True)

class CheckedImports(importlib.abc.MetaPathFinder):
    def __init__(self, root, manifest):
        self.root, self.manifest = root.resolve(), manifest

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        expected_paths = []
        for scope in self.manifest['source_roots']:
            for suffix in ['.py', '/__init__.py']:
                relative = scope.rstrip('/') + '/' + fullname.replace('.', '/') + suffix
                if relative in self.manifest['files']:
                    expected_paths.append(contained(self.root, relative))
        if expected_paths and (spec is None or spec.origin is None or Path(spec.origin).resolve() not in expected_paths):
            raise ConsistencyError(f'Pinned local module is shadowed: {fullname}')
        if spec is None or spec.origin in (None, 'built-in', 'frozen'):
            return None
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(self.root):
            return None  # Third-party dependencies are outside this local-source gate.
        relative = origin.relative_to(self.root).as_posix()
        expected = self.manifest['files'].get(relative)
        if expected is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ConsistencyError(f'Unpinned repository import: {fullname} at {origin}')
        spec.loader = CheckedLoader(fullname, str(origin), expected)
        return spec

def resolve_binding(root, binding):
    module = importlib.import_module(binding['module'])
    target = module
    for name in binding['symbol'].split('.'):
        target = getattr(target, name)
    if not callable(target):
        raise ConsistencyError('Pinned binding is not callable')
    path = inspect.getsourcefile(target)
    if path is None or Path(path).resolve() != contained(root, binding['source']):
        raise ConsistencyError(f'Imported callable points to a different source: {binding}')
    return target

def execute(root, manifest_path, expected_digest, output):
    """Run only from a clean launcher process; runner must use the checked constructor."""
    root = root.resolve()
    if output.exists():
        raise ConsistencyError('Output already exists; refusing to overwrite a prior run')
    manifest = verify(root, manifest_path, expected_digest)
    # A source hash cannot authenticate code objects imported before this gate.
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if file and Path(file).resolve().is_relative_to(root):
            if Path(file).resolve() != Path(__file__).resolve():
                raise ConsistencyError(f'Repository module already loaded before verification: {name}')
        if file and Path(file).resolve() != Path(__file__).resolve():
            for scope in manifest['source_roots']:
                if any(scope.rstrip('/') + '/' + name.replace('.', '/') + suffix in manifest['files'] for suffix in ['.py', '/__init__.py']):
                    raise ConsistencyError(f'Pinned module already imported before verification: {name}')
    finder = CheckedImports(root, manifest)
    old_path = list(sys.path)
    sys.path[:0] = [str(contained(root, p)) for p in manifest['source_roots']]
    sys.meta_path.insert(0, finder)
    receipt = dict(schema=1, model_id=manifest['model_id'], manifest_sha256=expected_digest,
                   constructor=manifest['constructor'], runner=manifest['runner'],
                   constructor_kwargs=manifest.get('constructor_kwargs', {}),
                   runner_config=manifest.get('runner_config', {}),
                   python=sys.version, executable=sys.executable,
                   started_utc=datetime.now(timezone.utc).isoformat(), status='verified', constructor_calls=0)
    try:
        constructor = resolve_binding(root, manifest['constructor'])
        runner = resolve_binding(root, manifest['runner'])
        verify(root, manifest_path, expected_digest)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'approved_manifest.json').write_bytes(manifest_path.read_bytes())
        write_json(output / 'source_receipt.json', receipt)
        def construct():
            verify(root, manifest_path, expected_digest)
            instance = constructor(**manifest.get('constructor_kwargs', {}))
            receipt['constructor_calls'] += 1
            write_json(output / 'source_receipt.json', receipt)
            return instance
        runner(constructor=construct, output_dir=output, config=manifest.get('runner_config', {}))
        if receipt['constructor_calls'] == 0:
            raise ConsistencyError('Runner did not call the pinned constructor')
        verify(root, manifest_path, expected_digest)
        receipt['status'] = 'complete'
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'source_receipt.json', receipt)
    except BaseException as error:
        if (output / 'source_receipt.json').exists():
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            write_json(output / 'source_receipt.json', receipt)
        raise
    finally:
        sys.meta_path.remove(finder)
        sys.path[:] = old_path

def require_registered_runner(source):
    raise ConsistencyError(
        f'Direct training is disabled for {Path(source).name}. '
        'This training runner is not a direct entry point. '
        'Use source_consistency.py run with an independently approved source pin; '
        'do not generate a pin automatically to bypass this check.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Create an UNAPPROVED candidate; never updates the registry')
    prepare.add_argument('--spec', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    run = commands.add_parser('run')
    run.add_argument('--model-id', required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.output.exists():
            raise ConsistencyError('Candidate output already exists')
        spec = json.loads(args.spec.read_text(encoding='utf-8'))
        spec.update(schema=1, files=inventory(ROOT, spec['source_roots']))
        write_json(args.output, spec)
        print('UNAPPROVED candidate SHA-256:', digest(args.output))
        return
    registry = json.loads(REGISTRY.read_text(encoding='utf-8'))
    pin = registry['models'].get(args.model_id)
    if pin is None:
        raise ConsistencyError('Model identity has no approved source pin; refusing to run')
    manifest_path = contained(ROOT, pin['manifest'])
    manifest = verify(ROOT, manifest_path, pin['sha256'])
    if manifest['model_id'] != args.model_id:
        raise ConsistencyError('Registry model identity differs from manifest')
    execute(ROOT, manifest_path, pin['sha256'], args.output)

if __name__ == '__main__':
    main()
