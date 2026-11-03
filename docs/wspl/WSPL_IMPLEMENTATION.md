# WSPL implementation gates

November 3, 2026. Planned work, not implemented APIs. See EXPERIMENT_PROGRAM.md
and the archived supplied proposal. Keep this as a language component within
Skillforge until there is a concrete reason to split repositories.

Current progress: the scalar JSON IR/type reader and exact reference implement
an initial part of G0. The fresh dense ReLU compiler/independent artifact loader
in COMPILER.md implement an initial G1, including optional Torch emission when
coefficients are exactly representable. Text parsing, broader types and
subsequent gates remain incomplete. VERIFICATION.md now
describes initial G2 numeric I/O contract checks for serialized real graphs, with
non-vacuity and exact counterexample replay, and an initial compiler-independent
source-to-artifact equivalence check over the declared input box (real
arithmetic only; no IEEE execution or proof-carrying certificate). INTERVENTION_EVIDENCE.md separately
implements empirical opaque-network activation probes; these never become proof
statuses. The known-program solver checks are not scientific synthesis trials,
IEEE execution proofs or physical guarantees.

## G0: typed core and independent interpreter

Implement named bounded inputs, units/normalization, shapes, exact rational
constants, acyclic expressions and affine/ReLU/clip/min/max/static wiring.
Truth is not probability; Boolean contracts and executable graded expressions
remain distinct. Weighted Lukasiewicz blocks can be expressed by supported
affine/clamp operations. Reject unsupported discontinuous branches, variable
multiplication/division and dynamic loops rather than tracing them accidentally.

Start with one explicit module form from the proposal, not every proposed
extension. An independent Fraction-based reference evaluates exact rational
inputs without importing the compiler or neural runtime. Include dimensional
and shape failures, duplicate bindings, algebraic cycles and non-idempotent
graded conjunction counterexamples in engineering tests.

Text front end (issue #3): PARSER.md parses the proposal's BrakePatch module form
and Core PWA expressions into this JSON IR and contract suites. The proposal text
compiles to its stated weights and its contracts prove on the compiled artifact.
The `normalized_*` unit reading is open for review; tensors, functions, stateful
modules and baseline/edited `preserves` remain unimplemented.

## G1: compiler and artifact

Compile BrakePatch and ApproachP (the inspected proportional standoff controller)
into small ReLU graphs with deterministic source maps. Emit/read non-executable
weights and immutable graph/I/O metadata. Fresh graph and fixed-architecture
allocation are separate modes; no silent width/depth expansion. Preserve a
canonical DAG so repeated subexpressions and signed relays are explicit.

Targets are rational mathematical IR and a separately named deployed numerical
artifact. Verify against the actual serialized stored coefficients, interpreted
as exact dyadic rationals where appropriate. Arbitrary rational source constants
may not be exactly representable as floating-point weights. Exact-only compilation
must prove equality after lowering/serialization or reject/request an explicitly
supported approximation mode; ordinary float casting is not an equivalence proof.

Boundary probes and mutation tests catch engineering errors but remain empirical.
Keep semantic lowering lemmas and serialized-graph equivalence separate. The
current RMV export/reference is useful antecedent code, not a WSPL compiler.
RMV fixed-capacity compilation begins with the representable subset only.

## G2: independent verifier spike

The initial verifier now uses Z3/QF_LRA for small serialized ReLU graphs, with
the engineering checks and limits in VERIFICATION.md. Exact Boolean enumeration
and full source equivalence remain separate extensions. Z3's mathematical real
sort is distinct from IEEE floats
([official arithmetic guide](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/)).
Do not infer deployed FP32/FP64 guarantees from real-arithmetic UNSAT.

Check domain feasibility before universal implication claims. Query the negation
of each contract over the serialized graph: UNSAT supports PROVED under the
recorded solver/encoding trust model; SAT needs exact reference witness replay;
timeouts/unsupported encodings are UNKNOWN, malformed inputs INVALID. Report
these statuses per contract, not one vague safe flag. Trusted-solver results are
not independently proof-carrying certificates unless proof validation is built.

Required regressions: all three BrakePatch contracts, actuator/sign/linear-region
contracts for ApproachP, deliberately reduced braking gain, wrong input order,
stale artifact hash, vacuous domain, conflicting assumptions, unsupported operator
and resource exhaustion. Failed random search is never a proof. No physical
collision-avoidance theorem is implied by a braking command property.

## G3: independently trained imports and transactional patches

Import a small supported learned MLP with full tensor orientation, activations,
named preprocessing/postprocessing, dtype/operator profile and baseline hash.
Support a matched output-patch capacity first; verify the whole composed graph
and explicit preservation domain. Report augmentation as augmentation. In-place
mask-limited edits are a separate supported target and later experiment track.

Artifact bundles bind source/IR, graph, weights, I/O and preprocessing, contracts,
verification semantics, parent hashes and source maps. Changing any bound item
invalidates affected claims. Keep original artifacts accessible and record
failures. Original learned networks must not be replaced with convenient
compiler-generated equivalents without separate evidence.

G3a progress (issue #6, IMPORT.md): frozen hidden-ReLU/final-linear tensors import
losslessly from a hex-float bundle into an `imported` ReLUArtifact whose layers are
exactly the tensors, and relational parent/edited contracts (preservation regions,
tolerances, relative repair bounds) are checked in QF_LRA with replayed witnesses.
Pre/postprocessing composition, output-patch composition and transaction
manifests/rollback (G3b) and any learned parent remain missing.

## G4: fair proposer/evaluator harness

Initial S1 kernel now exists: authoring_study.py implements all six finite-domain
arms at a common fixed capacity, data-only candidate parsing, feedback separation,
budgets and one-shot final selection. run_authoring_screen.py grades request/turn-
bound offline transcripts and preserves all 36 denominators. See
FORMAL_EXPERIMENTS_V1.md for the narrower front ends and finite proof scope.
Live isolated provider collection, billing receipts, pre-call budget enforcement
and sealed task access remain unimplemented; this does not complete G4.

Implement the six arms, immutable trial configuration, construction-code isolation,
independent proposer state, feedback-on/off firewall, shared candidate verifier,
bounded logging and final-data access restrictions. Audit actual billed tokens,
tool/candidate/verifier calls, environment steps and all feedback contents.
The existing scientific source gate pins code identity; it does not implement
this sandbox, blind data or make LLM calls. Those are real missing components.

Engineering release criteria include simulated proposer faults and attempts to
change contracts, leak final data, import evaluators or omit failed candidates.
Do not evaluate a new scientific method on engineering fixtures inspected while
developing that method.

## G5: discrete screen, then broaden

Only after G0-G4 can the 36-run discrete screen be concretely budgeted/pinned.
Its independent task generator, fresh instances, model/API identity, monetary
and wall caps, final evaluation counts, outcome thresholds and analysis manifest
remain to be set before approval. First use pilot results to diagnose feasibility
and cost; determine confirmation repetitions without promoting that pilot.

Next add pinned affine/PWA plant dynamics and finite-horizon uncertainty models.
Robot task families require their missing physical skills, observation interfaces,
independent observers and executor. Do not describe the sixteen catalog entries
as runnable or MuJoCo rollouts as universal closed-loop proofs. Transformer and
learned symbolic recovery profiles follow useful controller results.

## Immediate next implementation slice

Finish isolated provider collection and freeze model/tasks/cost for the narrow S1
finite-domain screen in FORMAL_EXPERIMENTS_V1.md. This pilot uses independent
finite artifact evaluation; it does not require or establish universal source
equivalence. G3 learned import/patches are prerequisites for S2 rather than a
requirement for this supplied-mechanism S1 pilot. General source/numerical proof,
script construction, S2 and S3 gates remain separate from S1 feasibility.
