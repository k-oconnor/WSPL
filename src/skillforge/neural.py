"""Small stateless motor blocks with explicit normalization and exact export.

Optional PyTorch module. RMV uses hard clamps and ordinary autograd, with
nonnegative logical weights projected after updates; no surrogate derivatives.
"""
import torch
from torch import nn
from .motor import CHANNELS
from .rmv import RMVReference

LOWER = (-.5,)*6 + (-1.,)*3 + (0., 0., 0.)
UPPER = (.5,)*6 + (1.,)*3 + (.08, 1., 1.)


class NormalizedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("lower", torch.tensor(LOWER))
        self.register_buffer("upper", torch.tensor(UPPER))

    def predicates(self, x):
        if x.shape[-1] != 12 or not torch.isfinite(x).all():
            raise ValueError("expected twelve finite input channels")
        return ((x-self.lower)/(self.upper-self.lower)).clamp(0, 1)


class MotorMLP(NormalizedBlock):
    def __init__(self, seed=7):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layers = nn.Sequential(nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 32),
                                        nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.layers(2*self.predicates(x)-1).clamp(-1, 1)

    def project_(self):
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("nonfinite MLP parameters")


class MotorRMV(NormalizedBlock):
    def __init__(self, seed=7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Asymmetric rows with sums in [1.2,1.8]: at predicate midpoint,
        # conjunctions are interior and distinct rather than dead/symmetric.
        w = torch.rand(56, 24, generator=g) + .1
        w *= (1.2 + .6*torch.rand(56, 1, generator=g))/w.sum(-1, keepdim=True)
        self.and_weights = nn.Parameter(w)
        v = torch.rand(4, 56, generator=g) + .1
        v *= .8/v.sum(-1, keepdim=True)
        self.or_weights = nn.Parameter(v)
        self.readout_weights = nn.Parameter(.3*torch.randn(4, 4, generator=g))
        self.readout_bias = nn.Parameter(torch.zeros(4))

    def internals(self, x):
        # Preserve FP32 parameters/optimizer, but evaluate the exported logical
        # arithmetic in FP64. Trained dense sums amplified FP32 rounding beyond
        # the declared 2e-6 scalar-fidelity tolerance. Casts remain differentiable;
        # this does not introduce surrogate gradients or change the hard graph.
        p = self.predicates(x.double())
        literals = torch.stack((p, 1-p), dim=-1).flatten(-2)
        clauses = (1-(1-literals) @ self.and_weights.double().T).clamp(0, 1)
        heads = (clauses @ self.or_weights.double().T).clamp(0, 1)
        commands = (heads @ self.readout_weights.double().T+self.readout_bias.double()).clamp(-1, 1).to(x.dtype)
        return dict(predicates=p, literals=literals, clauses=clauses, heads=heads, commands=commands)

    def forward(self, x):
        return self.internals(x)["commands"]

    @torch.no_grad()
    def project_(self):
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("nonfinite RMV parameters")
        self.and_weights.clamp_(min=0)
        self.or_weights.clamp_(min=0)

    def export(self):
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError("nonfinite RMV parameters")
        graph = dict(format="skillforge.rmv.v1",
                     inputs=dict(names=list(CHANNELS), lower=self.lower.detach().cpu().tolist(),
                                 upper=self.upper.detach().cpu().tolist()),
                     literal_sources=[dict(input=i, negated=n) for i in range(12) for n in (False, True)],
                     output_lower=[-1.]*4, output_upper=[1.]*4)
        for name in ("and_weights", "or_weights", "readout_weights", "readout_bias"):
            graph[name] = getattr(self, name).detach().cpu().tolist()
        RMVReference(graph)  # independent shape, finite-value and sign validation
        return graph


def reference_command(x, gain=20.):
    """Declared stateless teacher; position error in channels 3:6, open gripper."""
    if x.shape[-1] != 12 or not torch.isfinite(x).all():
        raise ValueError("expected twelve finite channels")
    return torch.cat(((gain*x[..., 3:6]).clamp(-1, 1), -torch.ones_like(x[..., :1])), -1)


@torch.no_grad()
def authored_approach(model, gain=20.):
    """Unblinded analytic weights for the reference P controller.

    No optimizer/simulator calls. Both architectures use their complete ordinary
    forward path. Unused capacity remains in the parameter count and is reported.
    """
    if not isinstance(model, (MotorMLP, MotorRMV)) or gain <= 0 or not torch.isfinite(torch.tensor(gain)):
        raise ValueError("supported block and positive finite gain required")
    for p in model.parameters(): p.zero_()
    if isinstance(model, MotorMLP):
        first, second, last = model.layers[0], model.layers[2], model.layers[4]
        for axis in range(3):
            first.weight[2*axis, 3+axis] = 1
            first.weight[2*axis+1, 3+axis] = -1
            second.weight[2*axis, 2*axis] = 1
            second.weight[2*axis+1, 2*axis+1] = 1
            last.weight[axis, 2*axis] = gain*.5
            last.weight[axis, 2*axis+1] = -gain*.5
        last.bias[3] = -1
    else:
        for axis in range(3):
            model.and_weights[axis, 2*(3+axis)] = 1
            model.or_weights[axis, axis] = 1
            model.readout_weights[axis, axis] = gain
            model.readout_bias[axis] = -gain*.5
        model.readout_bias[3] = -1
    return model
