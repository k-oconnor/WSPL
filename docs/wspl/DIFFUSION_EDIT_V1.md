# Diffusion coordinate editing: Python versus WSPL

Preparation only; fresh exact source approval required. Six Astra/xhigh one-shot
proposals, three per interface. Each proposal is one generic program applied to
all three released denoisers. The three networks are transfer checks of the same
algorithm, not independent author replicates. Order Python/WSPL for draw 0,
WSPL/Python for draw 1, Python/WSPL for draw 2. No feedback/retries/training.

Task: two-pixel rightward cyclic translation, with time coordinates unchanged.
Require `g(Px,t)=P f(x,t)`. Cropping is excluded because it is not invertible.
Same fixed 23,824-parameter architecture, all weights editable. Parent arrays
are accessible to both construction programs; no weight dumps/class labels are
needed for the generic transformation. Both arms receive the complete common
interface guide. Python is real NumPy code in bubblewrap, not literal edits.
One program execution per parent for each arm, no execution feedback to Astra.

This is a controlled coordinate/circuit-boundary edit with a known analytic
solution. It does not require reading learned semantic features, adding nodes,
or graph surgery. WSPL placement is available but optional; report actual syntax
usage. A ceiling supports feasibility only and calls for a harder follow-up.

Primary descriptive success: valid output for all three parents and numerical
agreement for every parent. Probe panel: 275 inputs (zero, +ones, -ones, 144
basis vectors, 128 uniform[-4,4] vectors), six times 0/1/249/499/749/999.
Probe RNG PCG64 seed9300000. Max absolute error tolerance .001.
Generation: 64 paired samples per parent, PCG64 seed9400000, all 1,000 reverse
steps with correctly permuted initial and incremental noise. Same .001 tolerance.
All samples archived, no filtering. Parent, candidate and target arrays/previews
saved. Same reconstructed posterior-variance sampler as the completed replay.
Prediction- and generation-errors reported separately; finite checks enforced.

Separately report an exact real-graph certificate when stored weights match the
algebraic permutation of the parent's first-layer input rows and last-layer
output columns/biases, retaining all other coefficients. This is a sufficient
structural witness, not a complete equivalence checker: alternative hidden-unit
bases can be correct without matching it. No claim of universal IEEE equivalence;
finite-panel success alone is not a universal functional proof.

Report all 6 attempts including invalids, numerical successes by arm, each of
three certificate outcomes per proposal, all provider usage, construction time,
sources, edit plans, parent/candidate hashes and paired samples. No superiority
test from three draws. Descriptive next step: both arms at least 2/3 complete
successes supports larger testing; 3/3 each is a ceiling. Otherwise diagnose
interface/authoring failures before increasing budget. Partial collection is
unresolved, never silently removed from denominators.

Uses existing Codex ChatGPT usage, unchanged one-shot collector, explicit
gpt-6-astra/xhigh. No enforceable dollar/output-token cap or API-key fallback.
20-minute call / four-hour overall guards. CPU-only construction/evaluation;
expected 30–90 minutes dominated by six author calls. No model calls before
approval. Clean stop checked between calls; collector checks while running.

Python sandbox: isolated user/PID/network/IPC/UTS namespaces, clear environment,
read-only runtime/input, bounded writable output file and tmpfs, no host home or
credentials. RLIMITs and 20s worker guard. Existing completed Kinetix runtime
unchanged; separate `/home/kevin/venvs/skillforge-author`. Tests include host,
network, parent-write and fork denial. Pin Windows and Linux runtime resources.
The Python source is untrusted data, never executed outside this sandbox.

Reference artifacts: upstream4d09ba6d6a7e891dd8063974dfe81b3d78e3494f,
exp2_diffusion. No original paper editing baseline exists for this task; this is
our extension. Oracle implementations and evaluator-only probes are excluded
from proposer context. No automatic follow-on Go experiment is launched.
