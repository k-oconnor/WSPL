"""Data-only submission boundary, independent reconstruction and freeze ledger."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from .wspl_lora import snapshot, identity, describe, source_for_pair, construct_native, construct_wspl
from .wsp_artifact import Bits, digest
from .wsp_values import pack
from .reliable_json import save


def safe_file(work, relative, maximum):
    work = Path(work).resolve(strict=True)
    rel = Path(relative)
    if rel.is_absolute() or rel.drive or '..' in rel.parts:
        raise ValueError('A relative workspace file is required')
    path = work/rel
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(work) or path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError('Linked or out-of-workspace submission')
    if path.stat().st_size > maximum:
        raise ValueError('Submission exceeds byte limit')
    return path


def load_arrays(path, maximum_elements=25_000_000):
    """Bound decompression/allocation before NumPy parses any tensor data."""
    if Path(path).stat().st_size > 128_000_000:
        raise ValueError('Array archive exceeds 128 MB')
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= 512 or len({m.filename for m in members}) != len(members):
            raise ValueError('Duplicate or excessive archive entries')
        if sum(m.file_size for m in members) > 110_000_000:
            raise ValueError('Excessive decompressed array bytes')
        count = 0
        for member in members:
            if not member.filename.endswith('.npy') or any(c in member.filename for c in '/\\:'):
                raise ValueError('Flat NPY archive required')
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version != (1, 0):
                    raise ValueError('NPY version 1 required')
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10000)
                if dtype != np.dtype('<f4') or not 1 <= len(shape) <= 2 or any(s < 1 for s in shape):
                    raise ValueError('Nonempty FP32 vectors/matrices required')
                elements = 1
                for s in shape:
                    elements *= s
                count += elements
                if count > maximum_elements or stream.tell()+elements*4 != member.file_size:
                    raise ValueError('Array capacity/header mismatch')
        with np.load(path, allow_pickle=False) as data:
            result = {k: np.array(data[k], dtype=np.float32, order='C', copy=True) for k in data.files}
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError('Nonfinite array')
    return result


def stage_submission(work, relative, out, arm, expected_parent):
    """Copy inert files out of a drained sandbox; never import author Python."""
    out = Path(out); out.mkdir()
    path = safe_file(work, relative, 2_000_000)
    request = json.loads(path.read_text(encoding='utf8'))
    required = {'parent_id', 'arrays', 'note'} if arm == 'native' else {'parent_id', 'programs', 'bindings', 'note'}
    if not isinstance(request, dict) or set(request) != required or request['parent_id'] != expected_parent:
        raise ValueError('Submission schema or original parent identity mismatch')
    if not isinstance(request['note'], str) or len(request['note']) > 4000:
        raise ValueError('Describe the method in note (at most 4000 characters)')
    field = 'arrays' if arm == 'native' else 'bindings'
    if request[field] is not None:
        archive = safe_file(work, request[field], 128_000_000)
        target = out/(field+'.npz')
        target.write_bytes(archive.read_bytes())
        request[field] = target.name
    elif arm == 'native':
        raise ValueError('Native arrays required')
    if arm == 'wspl' and (not isinstance(request['programs'], dict) or not request['programs']
                         or len(request['programs']) > 196 or not all(isinstance(v, str) for v in request['programs'].values())):
        raise ValueError('WSPL projection-to-body mapping required')
    save(out/'submission.json', request)
    return out/'submission.json'


def reconstruct(parent, submission, arm):
    path = Path(submission)
    request = json.loads(path.read_text())
    if request['parent_id'] != identity(parent):
        raise ValueError('Stale parent')
    if arm == 'native':
        candidate = snapshot(load_arrays(path.parent/request['arrays']))
        if candidate.keys() != parent.keys():
            raise ValueError('Complete original adapter bank required')
        replacements = {p: {f: candidate[p+'.lora_'+f] for f in ('A', 'B')}
                        for p in describe(parent)['projections']}
        candidate, receipt = construct_native(parent, replacements, expected_parent_id=request['parent_id'])
    elif arm == 'wspl':
        arrays = load_arrays(path.parent/request['bindings']) if request['bindings'] else {}
        import re
        if any(not re.fullmatch('[A-Za-z][A-Za-z_0-9]*', k) or k in ('A', 'B', 'adapter', 'candidate') for k in arrays):
            raise ValueError('Invalid binding identifier')
        bindings = {k: np.array([Bits(int(v)) for v in a.view(np.uint32).flat], dtype=object).reshape(a.shape)
                    for k, a in arrays.items()}
        catalog = describe(parent)['projections']
        programs = {}
        for p, body in request['programs'].items():
            if p not in catalog or len(body.encode()) > 1_000_000:
                raise ValueError('Unknown projection or excessive source')
            # Declare referenced constants; the bridge still archives the shared binding bank.
            used = {k: v for k, v in bindings.items() if re.search(r'\b'+re.escape(k)+r'\b', body)}
            declaration = '\n'.join(f'const {k}=tensor("{k}",sha256="{digest(pack(v))}");' for k,v in used.items())
            programs[p] = source_for_pair(catalog[p]['pair_id'], body).replace('edit adapter', declaration+'\nedit adapter', 1)
        candidate, receipt = construct_wspl(parent, programs, expected_parent_id=request['parent_id'], bindings=bindings)
        receipt['expanded_programs'] = programs
    else:
        raise ValueError('Unknown author arm')
    changes = {}
    for p in describe(parent)['projections']:
        if any(not np.array_equal(parent[p+'.lora_'+f].view(np.uint32),candidate[p+'.lora_'+f].view(np.uint32)) for f in ('A','B')):
            a0,b0 = (parent[p+'.lora_'+f].astype(np.float64) for f in ('A','B'))
            a1,b1 = (candidate[p+'.lora_'+f].astype(np.float64) for f in ('A','B'))
            a,b = np.concatenate([a1,a0]),np.concatenate([b1,-b0],axis=1)
            changes[p] = float(np.sqrt(max(0., np.sum((b.T@b)*(a@a.T)))))
    receipt.update(method_note=request['note'], effective_unscaled_delta_norm=changes,
                   attribution='Audit all Python calls; WSPL serialization alone does not establish synthesis benefit.')
    return candidate, receipt


def freeze_ledger(candidates, path):
    if set(candidates) != {'parent','gradient','native0','wspl0','wspl1','native1','native2','wspl2'}:
        raise ValueError('All eight arms must finish before final evaluation')
    rows = {}
    for arm, candidate in candidates.items():
        if candidate is None:
            if arm in ('parent','gradient'):
                raise ValueError('Control artifact missing')
            rows[arm] = dict(status='authoring_failed', file=None)
        else:
            candidate = Path(candidate).resolve(strict=True)
            rows[arm] = dict(status='frozen', file=str(candidate),
                file_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                adapter_id=identity(load_arrays(candidate)))
    save(path, rows)
    return rows


def verify_frozen(row):
    if row['status'] != 'frozen' or hashlib.sha256(Path(row['file']).read_bytes()).hexdigest() != row['file_sha256']:
        raise ValueError('Frozen artifact changed')
    state = load_arrays(row['file'])
    if identity(state) != row['adapter_id']:
        raise ValueError('Frozen factor identity changed')
    return state
