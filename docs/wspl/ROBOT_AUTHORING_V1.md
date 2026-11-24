# Matched robot LoRA authoring v1

Prepared November 24, 2026. This is the concrete successor to the proposed
SUPERVISION_WSPL_LORA_V1.md. No hosted author requests, acquisition missions,
adaptation training or final evaluations have launched. An exact separately
approved source digest is required by AGENTS.md.

## Question and common deployment

Can Astra construct a useful persistent adapter repair, and does adding WSPL
help compared with ordinary Python construction? This is an exploratory screen,
not a population-level efficiency or general capability demonstration.

All conditions use Qwen2.5-Coder-1.5B-Instruct, the frozen ordinary-practice parent
from fresh-robot-parent-v2, 196 rank16/alpha32 LoRA projection pairs (18,464,768
FP32 coefficients), the existing raw robot codec and full-vocabulary one-token
greedy decoding. Base weights, rank, scaling, prompts, motors, sensors and
mission semantics cannot change. No author, external solver, Python program,
new hook or learned router runs at deployment. The supplied nominal and inverse
calibration motor blocks are analytic WSPL compilations, not rediscovered motors.

Parent file SHA256:
`2cd5bd6662db5bdb39e6d341250abfca083672e2b9794c939abc145b49470276`.
Parent factor SHA256:
`431806f5f4fc6788969a0f954d8fe594caa7b8597b3b293c2d23f57e61459c75`.
Frozen base SHA256:
`540b6c0285e9d26362156163cef352c1a27657dc7b0372d0c36404a6c5baddc1`.
Prior readiness was 16/16 ordinary successes versus 0/16 before training; this
does not establish changed-calibration adaptation or the new final baseline.

## Arms and data

Eight arms: unchanged parent, fresh gradient control, three native Python author
sessions and three Python-plus-WSPL sessions. Author order is native0, WSPL0,
WSPL1, native1, native2, WSPL2. Sessions receive independent workspaces and no
other author's replies, candidate weights or scores. Gradient-control weights
and outcomes are never supplied. Model sampling is not seed-controlled.

The four physical conditions are ordinary, changed from the start, switch after
the first waypoint, and return to nominal after the first waypoint. For condition
index c, seed is the panel seed plus1000*c+i; waypoint count is3+i%3. Every mission
allows32 executive decisions. The motor/physics limits remain those of the parent.

- Acquisition: panel seed73300000, four missions per condition,16 total. A supplied
  teacher uses only public displacement/error feedback to probe, select, reach and
  stop. If any acquisition mission fails, stop before fitting or author requests.
- Development: seed73400000, four per condition,16 total. Score the original
  parent once; each accepted proposal gets the same panel and full public decision
  records, never private simulator state. No candidate selection uses final data.
- Final: seed73500000,16 per condition,64 total. All controls and author outcomes
  are recorded in an immutable hash ledger before any final simulator opens.
  With six valid authors there are512 final episodes. Invalid author sessions stay
  failures and receive no silently substituted policy or evaluation.

These seeds are disjoint from each other and the parent's training/readiness
seeds. They sample the same engineered reaching domain; do not claim real-world
robot transfer, independent natural tasks, or broad language retention.

Every author receives all acquisition prompts, action labels and public states;
all parent factors; the codec; and acquisition-only diagnostics: final-block
down-projection inputs X, pre-final-RMSNorm residuals R, nine action logits L,
one-hot labels Y, those nine frozen output-head rows W and RMSNorm scale N.
They also receive parent development counts. No full base checkpoint is exposed
inside the author workspace. Both interfaces get identical numerical information.

The gradient control starts from the same parent. It performs384 AdamW updates
at peak3e-5, weight decay.01, warmup16 followed by cosine decay to10%, norm clip1,
and eight one-token targets/update: two sampled decisions from each acquisition
condition. Schedule seed114991. It uses all196 LoRA pairs, preserves the base,
and saves a rolling optimizer/RNG checkpoint every1800seconds and at completion.
The control uses3072 target presentations. It is not compute- or feedback-matched
to the authors; no efficiency claim can be based on this control alone.

## Author interface and isolation

Each author gets16 Astra/xhigh responses, at most four evaluation attempts, and
the same full Python/NumPy/PyTorch/CUDA/Transformers runtime. A response is one
JSON request for Python, evaluate or finish. Malformed responses count. Malformed
or invalid evaluation submissions count against the four attempts. Python code
and stdout, all tool responses, author messages and raw collection usage are saved.

Each response uses a fresh zero-native-tool Codex CLI collection with the full
visible transcript supplied in the prompt. This is an author session reconstructed
from visible history, not a persistent private reasoning thread. Actual offline
prompt checks confirm both profiles expose zero CLI tools. The delegated Python
tool runs separately in Windows AppContainer; its write permission is explicit.

The native arm submits complete FP32 factor arrays constructed with ordinary
Python. Its separate runtime contains identical Python/numerical packages but no
WSPL package; its profile cannot read the WSPL runtime. Attempts
to reconstruct/use WSPL indirectly are also outside the native-arm instructions.
The WSPL arm submits one fixed exact-to-storage body per selected projection,
with optional hash-pinned FP32 construction bindings. Trusted code adds original
pair hashes and replays every program; only reconstructed arrays can deploy.
Unedited pairs retain original bits. Both arms must preserve all factor names,
shapes, dtype, rank and scale and pass the same finite-value checks.

The common host's array validation/identity bookkeeping uses the existing bridge
in both arms. This does not give the native author access to the WSPL language.
WSPL binding construction in Python is allowed; copying precomputed factors
through WSPL is serialization and must not be described as WSPL-driven synthesis.

Each author profile can read only its supplied inputs/runtime and permitted
Windows system resources, write its own persistent workspace, and cannot access
the final evaluator, other authors or private host files. There are no network
capabilities or inherited handles. Child processes inherit restrictions and are
terminated before host output collection. This shares the Windows kernel/GPU;
it is not a VM, and the process memory cap is not a VRAM quota. Sessions run serially.

Primary scope is direct construction, including disclosed analytic supervised
linear algebra. Iterative fitting/gradient optimization is not permitted in this
primary scope. Full Python cannot enforce that semantic rule: source audit must
classify violations as training-assisted and retain them visibly. Results are
marked pending method audit until all Python and WSPL construction is reviewed.

Each valid proposal becomes the last accepted candidate, regardless of whether
development scores improve. Finish or the response cap freezes that candidate.
No valid proposal means authoring failure. No best-on-final selection is possible.

## Budgets, stopping and measures

Maximum96 author responses,20minutes each; Python calls at most300seconds;
independent reconstruction at most180seconds. Full Python jobs allow16 processes
and16GiB aggregate committed process memory. Array archives are at most128MB and
25M FP32 elements; WSPL bodies at most1MB per pair, with existing interpreter
limits plus64M aggregate arithmetic operations and100K statements/loop iterations.

Development wall guard24hours, aggregate measured development execution budget
8hours (acquisition/diagnostics/control/Python/validation/development scoring),
and final guard6hours. The execution budget is checked between operations;
an already running bounded Python/validation operation may finish before the
overrun is detected. Author response wait time is recorded separately and counts
toward development wall time. Outer watchdog is30hours plus5minutes cleanup.
Existing Codex usage; no hard dollar or output-token cap. These are maximum
envelopes, not runtime promises. There is no model/API-key fallback.

Transport, simulator or infrastructure failure stops without automatic retry.
No final-score-driven repairs or automatic follow-on. A controlled exit/source
receipt is not by itself evidence that the experiment finished successfully.

Archive every episode, full-vocabulary token/action, invalid output, physical
trace, paired initial-state hash, proposal rejection, Python program, WSPL body,
binding file, factor identity and effective unscaled B*A difference. Keep accepted
construction artifacts and one rolling training checkpoint, without per-proposal
base copies. Report author time, Python/compilation/evaluation time, call/token
usage and interface validity separately. Do not treat episodes as author draws.

Predeclared per-candidate exploratory gate: parent ordinary score>=12/16;
candidate ordinary>=14/16 with at most one ordinary regression; at least eight
net corrections across the48 adaptation missions; and no condition-level net
loss. Report actual paired counts even when the gate fails. Each interface has
three author-session denominators, including failed authors. Gradient comparison
is descriptive. No general WSPL superiority or retention claim follows from this
small domain screen alone.

## Engineering evidence

All52 focused checks pass. Full18.5M-factor native/WSPL reconstruction and synthetic Qwen forward parity
pass, including finite synthetic backward, optimizer checkpoint reload, unchanged
base and exact original restoration (about3.24GiB peak allocated CUDA memory),
including the actual acquisition-diagnostic hooks on synthetic token sequences.
The real AppContainer/full-Python/submission-worker path passes with canned
responses, zero hosted calls or task predictions, final-data canary denial,
cross-author denial and native WSPL-package denial. All four supplied-teacher
engineering fixtures succeeded. Fake orchestration tests cover failure accounting,
no transport retry, all-arm freeze before final evaluation, and absent failed
authors. Exact fixture receipts are bound into the candidate source manifest.

The source manifest inventories src, experiments, tools and tests, all historical
learner/simulator/model resources, both independent author runtime copies,
the frozen parent, CLI/catalog/instructions and this protocol. Preparation never
updates approved.json. Approval and registration are separate actions.

An initial per-profile deny-ACL attempt on a shared runtime failed the native
WSPL-import test; it was not used for a scientific run. The final design uses
physically separate runtime copies and a default-denied cross-runtime boundary.
The failed fixture and source history are retained locally.
