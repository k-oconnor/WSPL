"""CPU replay components for the released CSAIL MinAtar Breakout policy.

Native tanh/float32 inference, empirical evaluation only. No WSPL proof, training,
provider calls or implicit execution of the downloaded author response.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

TENSORS = {
    'actor/dense1/kernel': (600,32), 'actor/dense1/bias': (32,),
    'actor/dense2/kernel': (32,32), 'actor/dense2/bias': (32,),
    'actor/logits/kernel': (32,6), 'actor/logits/bias': (6,),
}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_identity():
    """Fingerprint scientific core sources/binaries; rendering is not used."""
    import numpy as np
    import minatar
    roots = {'numpy':Path(np.__file__).parent, 'minatar':Path(minatar.__file__).parent}
    libs = roots['numpy'].parent/'numpy.libs'
    if libs.exists(): roots['numpy.libs'] = libs
    files = {name+'/'+p.relative_to(root).as_posix():file_hash(p)
             for name,root in roots.items() for p in sorted(root.rglob('*'))
             if p.is_file() and p.suffix in ('.py','.pyd','.dll','.so')}
    versions = {d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    body = dict(python=sys.version, numpy=np.__version__, minatar=importlib.metadata.version('MinAtar'),
                package_versions=versions, files=files)
    body['sha256'] = hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return body


def encode_observation(state):
    import numpy as np
    s = np.asarray(state)
    if s.shape != (10,10,4) or s.dtype != np.bool_:
        raise ValueError('native Breakout Boolean 10x10x4 state required')
    x = np.zeros((10,10,6),dtype=np.float32)
    for dest,source in ((0,0),(2,3),(3,1),(4,2)):
        x[:,:,dest] = s[:,:,source]
    return x.reshape(600)


def probabilities(logits):
    import numpy as np
    values = np.asarray(logits,dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError('six finite logits required')
    p = np.exp(values-values.max())
    return p/p.sum()


def sample_action(p, uniform):
    import numpy as np
    p = np.asarray(p,dtype=np.float64)
    if (p.shape != (6,) or not np.isfinite(p).all() or (p<0).any() or
        abs(float(p.sum())-1)>1e-12 or not 0 <= uniform < 1):
        raise ValueError('valid categorical distribution and uniform in [0,1) required')
    return min(int(np.searchsorted(np.cumsum(p),uniform,side='right')),5)


class BreakoutPolicy:
    def __init__(self, weights_path, expected_sha256):
        import numpy as np
        path = Path(weights_path)
        if path.stat().st_size > 1_000_000 or file_hash(path) != expected_sha256:
            raise ValueError('released artifact identity differs')
        with zipfile.ZipFile(path) as archive:
            if sum(x.file_size for x in archive.infolist()) > 1_000_000:
                raise ValueError('weight archive expansion limit')
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files) != set(TENSORS): raise ValueError('weight keys differ')
            self.weights = {key:archive[key].copy() for key in TENSORS}
        for key,shape in TENSORS.items():
            a = self.weights[key]
            if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all() or (np.abs(a)>1e6).any():
                raise ValueError('invalid weight tensor '+key)
            a.flags.writeable = False
        self.artifact_sha256 = expected_sha256
        self.mode = 'released'

    def uniform_control(self):
        import numpy as np
        self.weights = {key:np.zeros(shape,dtype=np.float32) for key,shape in TENSORS.items()}
        for a in self.weights.values(): a.flags.writeable = False
        self.mode = 'uniform_zero_weights'
        return self

    def tensor_identity(self):
        return hashlib.sha256(b''.join(self.weights[k].tobytes() for k in TENSORS)).hexdigest()

    def logits(self, x):
        import numpy as np
        x = np.asarray(x,dtype=np.float32)
        if x.shape != (600,) or not np.isfinite(x).all(): raise ValueError('600 finite inputs required')
        w = self.weights
        h1 = np.tanh(x@w['actor/dense1/kernel']+w['actor/dense1/bias'])
        h2 = np.tanh(h1@w['actor/dense2/kernel']+w['actor/dense2/bias'])
        result = h2@w['actor/logits/kernel']+w['actor/logits/bias']
        if not np.isfinite(result).all(): raise ValueError('nonfinite inference')
        return result


def episode(policy, env, *, env_seed, action_seed, max_steps, should_stop=lambda:False):
    import numpy as np
    if type(max_steps) is not int or max_steps < 1: raise ValueError('positive step cap required')
    env.seed(env_seed)
    env.reset()
    rng = np.random.Generator(np.random.PCG64(action_seed))
    reward = 0.
    counts = [0]*6
    for step in range(max_steps):
        if step % 128 == 0 and should_stop():
            return dict(status='incomplete', steps=step, discarded_partial_return=reward)
        action = sample_action(probabilities(policy.logits(encode_observation(env.state()))),rng.random())
        counts[action] += 1
        value,terminal = env.act(action)
        if not np.isfinite(value): raise ValueError('nonfinite environment reward')
        reward += float(value)
        if terminal: break
    return dict(status='complete', env_seed=env_seed, action_seed=action_seed,
                reward=reward, steps=step+1, terminal=bool(terminal),
                truncated=not bool(terminal), action_counts=counts)


def summarize(records):
    import numpy as np
    if not records or any(r['status']!='complete' for r in records):
        raise ValueError('only nonempty completed episodes may be summarized')
    values = np.array([r['reward'] for r in records],dtype=np.float64)
    return dict(episodes=len(records),mean=float(values.mean()),
                se=None if len(records)<2 else float(values.std(ddof=1)/np.sqrt(len(records))),
                min=float(values.min()),max=float(values.max()),
                mean_length=float(np.mean([r['steps'] for r in records])),
                truncations=sum(r['truncated'] for r in records))
