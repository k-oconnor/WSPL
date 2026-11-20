"""Two actual author requests must demonstrate disk persistence before research."""
import json
from pathlib import Path
import secrets
import numpy as np
from .supervisor_session import parse_request
from .supervisor_workspace import digest,save


def run_gate(root,workspace,budget,bridge):
    root=Path(root);root.mkdir(exist_ok=False);workspace=Path(workspace)
    target=workspace/'persistence_gate'
    if target.exists():raise ValueError('Persistence gate must start fresh')
    nonce=secrets.token_hex(16);history=[]
    requests=[
        'Before research, demonstrate that you can persist neural weights. Return a CPU Python request '
        'with at most 60 seconds. Create /work/persistence_gate; save float32 weights [[1,2],[3,5]] '
        'as array named weights in weights.npz using numpy.savez, and write receipt.json '
        'with the field nonce equal to '+nonce+'. Do not train or load Qwen. These writes are authorized.',
        'In this NEW Python process, load /work/persistence_gate/weights.npz from disk with numpy.load, '
        'multiply the weights matrix by float32 vector [2,3], and write reload.json in that directory '
        'with nonce '+nonce+', output as a JSON list, and weights_sha256 as the SHA-256 of the saved '
        'weights.npz bytes. Leave weights.npz unchanged. Return a CPU Python request of at most 60 seconds.'
    ]
    expected_hash=None
    for stage,instruction in enumerate(requests):
        folder=root/f'stage_{stage}';folder.mkdir()
        history.append(dict(role='user',content=instruction));budget.reserve_call()
        response=bridge.collect(instruction,history[:-1],folder)
        (folder/'response.txt').write_text(response,encoding='utf8')
        request=parse_request(response)
        if request['tool']!='python' or request['gpu'] or request['seconds']>60:
            raise ValueError('Persistence gate only allows a CPU Python job up to 60 seconds')
        outcome=bridge.execute(request,folder)
        history.extend([dict(role='assistant',content=response),dict(role='tool',content=outcome)])
        if outcome['outcome']!='complete':raise ValueError('Persistence job did not complete')
        # The host independently reads exported bytes, not the model's claims.
        if not target.resolve().is_relative_to(workspace.resolve()):raise ValueError('Gate path escaped workspace')
        weights=target/'weights.npz'
        for name in ('weights.npz','receipt.json')+ (('reload.json',) if stage else ()):
            p=target/name
            if p.is_symlink() or p.stat().st_size>16384:raise ValueError('Invalid persistence file')
        with np.load(weights,allow_pickle=False) as archive:
            w=archive['weights']
            if w.dtype!=np.float32 or not np.array_equal(w,[[1,2],[3,5]]):raise ValueError('Weight export differs')
        if json.loads((target/'receipt.json').read_text())['nonce']!=nonce:raise ValueError('Nonce mismatch')
        if stage==0:expected_hash=digest(weights)
        else:
            reload=json.loads((target/'reload.json').read_text())
            if reload!={'nonce':nonce,'output':[8,21],'weights_sha256':expected_hash} or digest(weights)!=expected_hash:
                raise ValueError('Reload did not reproduce saved weights')
    save(root/'receipt.json',dict(status='passed',author_calls=2,weights_sha256=expected_hash,
         independent_export_verified=True,separate_process_reload_verified=True))
    return history
