# WSPL: Weight Space Programming Language

**A typed programming interface for LLM-authored neural programs.**

WSPL investigates whether an LLM can construct, inspect and repair useful neural
controllers through programs with explicit semantics, rather than manipulating
opaque tensors alone. The intended executable artifacts are neural weights.
**Skillforge** is the robotics and reusable-skill testbed in this repository.

This is an early research prototype led by Kevin O'Connor. We distinguish a
program matching its weights, a controller satisfying an input-output contract,
and a robot satisfying a trajectory property. Each needs its own evidence.

## Current research status

**November 16, 2026 — neural control DSL prototype:** typed `move_to`, `grasp`,
`release` and reusable `transfer` programs now execute through frozen neural motor
and predicate blocks in a multi-object kinematic workcell. Explicit skill lifetimes,
independent completion checks and per-tick traces make composition inspectable.
64 engineering checks pass; supplied success and jammed-grasp fixtures are included.
No learned executive or physical-robot result is claimed. See
[design, limitations and reproduction](docs/wspl/NEURAL_CONTROL_DSL_V1.md).

**November 9, 2026:** the SmolLM capability and retention experiments are complete
and archived. Gradient-trained spatial adapters learned the bounded grid task,
but substantially degraded prose and ARC science performance when enabled.
Astra partially reduced that interference by scaling adapter outputs; it did not
rewrite the learned computation or use WSPL syntax in that repair. These studies
do not establish a WSPL advantage or general language-model capability programming.
No experimental adapter or repair is promoted into the supported design.
See the [current research handoff](CURRENT_RESEARCH.md) and
[completed results](docs/wspl/SMOLLM_EXPERIMENTS_2026-11-09.md).

## Current status

The WSP editor implements direct FP32 tensor editing, explicit affine/ReLU circuit
placement, immutable plans and frozen-artifact descriptions/checks. It now also
supports [graph surgery](docs/wspl/WSP_GRAPH_SURGERY.md): replace subgraphs,
transplant modules, rewire edges, add residual paths and untie shared weights in
stateless vector DAGs. Plans expose every boundary consumer and shared parameter.
The broader [tooling design](docs/wspl/WSP_TOOLING_DESIGN.md) and
[syntax](docs/wspl/WSP_SYNTAX_0_2.md) retain future extensions. The new frontend
preserves historical Core PWA semantics; engineering checks and completed
authoring screens are summarized under [Experiments](#experiments).

| Component | Status |
| --- | --- |
| WSPL 0.2 fixed-architecture editor | First pass: direct FP32 tensor construction, placed affine/ReLU circuits, plan/apply/inspect/diff/describe/check/probe CLI and APIs |
| WSPL 0.2 graph surgery | Stateless vector DAG replacement, transplantation, rewiring, residuals and explicit shared-weight separation; replayed plans and full edited-graph checks |
| Typed scalar JSON IR and exact rational interpreter | Implemented; engineering examples and negative checks |
| WSPL text front end (proposal module form) | Initial parser to JSON IR and contract suites; Core PWA only |
| Scalar IR to ReLU MLP weights, source map and independent loader | Initial implementation; fresh architecture only |
| Small real-graph I/O contract verifier | Initial Z3 prototype; non-vacuity and rational witness replay |
| Source IR to stored-weight equivalence check | Initial Z3 prototype; compiler-independent encoding, replayed differences |
| Numerical (IEEE) execution proof | Planned |
| Frozen MLP tensor import and parent/edited relational contracts | Initial G3a prototype; exact hex-float import with pre/postprocessing relays, Z3 relational checks |
| Output-patch composition and transactions | Initial G3b prototype; augmentation with verified slices, COMMITTED/REJECTED/UNRESOLVED gating, replayable manifests |
| Opaque-network intervention evidence | Paired activation probes, controls, replay and hash-chained empirical logs |
| Fresh LLM authoring | Native and matched native/WSPL Breakout screens completed; feedback-driven authoring remains planned |
| Small-alphabet LTLf compiler and strong finite-model planner | Implemented for supplied symbolic effect models |
| Hierarchical task expansion | Implemented at a supplied macro-summary cut; no physical primitive executor |
| Neural MLP/RMV blocks and single-tick LIBERO bridge | Implemented; independent sampled outcome observers |
| Pregrasp positioning pilot | Completed; scoped results below |
| Diverse robot tasks | 16 inspected specifications over 7 audited scene assets; contact/obstacle integration pending |

The proposal's first module form parses from text into the scalar JSON IR and
contract suites ([text front end](docs/wspl/PARSER.md)). The separate WSPL 0.2
editor implements the tensor/circuit and graph subsets linked above. The scalar
JSON IR compiles neural weights and supports scoped
real-graph contract checks. There is
no numerical execution proof. The available compiler command is documented below;
the proposal's general `wspl` command suite remains planned.

Try the supplied graph-edit example after installing the editor and verifier:

```sh
python -m pip install -e ".[editor,verify]"
python tools/demo_wsp_graph.py --output work/wsp_graph_surgery_demo
```

Choose a new output directory. The [example](examples/wsp_graph/README.md) replaces
one branch, adds a residual and separates a shared weight in three independent
candidates. It checks each intended change and preserved branch against the
actual stored graph. These are known-answer fixtures, not learned robot results.

The [graph stress checks](docs/wspl/WSP_STRESS_V1.md) now cover 100 seeded DAGs,
1,200 edit steps and preserved execution of three archived controller artifacts.
The [matched repair screen](docs/wspl/REPAIR_SCREEN_V1_RESULTS.md) completed:
**6/6 successes for direct editing and 6/6 with WSPL available**, across three
supplied circuit tasks and two fresh Astra draws each. All repair/preservation
claims and numerical checks passed. WSPL used `untie` in two proposals and raw
graph edits in four; general region replacement was not exercised. This is a
feasibility pass with a ceiling, not a comparative advantage or learned-controller
result. The twelve replies and independently replayed completion audit are archived.

The [reworked Breakout comparison](docs/wspl/BREAKOUT_REWORKED_V1_RESULTS.md)
completed and passed its descriptive feasibility rule. Three fresh WSPL policies
averaged **391.685** reward versus **370.516** for three native-edit policies on
the same original 20,486-parameter tanh network. All six were valid, with 1,000
paired episodes each. Across-proposal SDs were 125.2 and 120.0, respectively:
this small screen does not establish superiority. All WSPL proposals used
construction loops and exact arithmetic; their stored source was 56% shorter
and reported output-token use 20% lower in this run. Full plans, weights,
equations, scores and trajectories are retained, with
[all replies and sources archived](artifacts/wsp_breakout_v1/README.md).
The audit reconstructed every candidate and checked 6,000 episode records and
127,378 recorded actions. The subsequent [Kinetix repair screen](docs/wspl/KINETIX_REPAIR_V1_RESULTS.md)
also completed: all three native and three WSPL repairs succeeded in 1,024/1,024
episodes each, versus 0/1,024 for the parent. This is a success ceiling, not a
comparative performance advantage; all WSPL proposals used literal edits.

## WSPL syntax: programs that construct weights

WSPL 0.2 declares a target network and an explicit edit transaction. The following
runnable construction illustration adapts coordinate-weight assignments from an
[actual Breakout proposal](artifacts/wsp_breakout_v1/draw_1_wspl.source.wspl).
It illustrates syntax; it is not the complete game-playing controller.

```wspl
wspl 0.2;
target base=dense(widths=[600,32,32,6],activations=[tanh,tanh,linear],layout=in_out,storage=f32,init=zeros);

const gain=16;
edit base as candidate mode=fixed numeric=exact_to_storage {
  for r in range(10) {
    for c in range(10) {
      W1[6*(10*r+c)+3,15:20]=gain*[r,c,c,c,c];
      W1[6*(10*r+c)+4,0:2]=-10*gain*[c,r];
    }
  }
}
```

`W1` connects input rows to first-layer unit columns; `b1` is that layer's bias.
Here the input index names a grid cell and channel: channel 3 contains the ball,
and channel 4 its previous position. A few expressions populate coordinate-based
weights across the board. **Construction loops run once to produce weights.**
Inference executes the resulting fixed neural network, with no runtime loops or
policy code supplied by the author.

| Syntax | Meaning |
| --- | --- |
| `target base=dense(...)` | Declare shape, activations, layout, storage and zero initialization |
| `target base=artifact("name",sha256="...")` | Bind an existing artifact by exact identity; `...` must be replaced with its actual digest |
| `edit base as candidate mode=fixed` | Construct a new candidate without changing the parent's tensors or topology |
| `const gain=16;` / `for r in range(10) { ... }` | Named construction values and bounded expansion of repeated weight patterns |
| `W1[i,j]=v;` / `W2[0:4,0:4]=copy(W1[0:4,0:4]);` | Direct tensor writes, slices, broadcasts and statement-snapshot copies |
| `+=`, `-=`, `*=` | Update selected entries; later writes see the current candidate storage |
| `parent.W1[...]` | Read the original immutable tensor values |
| `numeric=exact_to_storage` | Exact rational construction arithmetic, rounded directly to FP32 at each assignment |
| `numeric=native_numpy_v1` | Historical native literal-edit arithmetic; this compatibility profile excludes loops and expressions |

Graph mode adds explicit operations on consumers and shared parameters. This
fragment was used successfully in the synthetic repair screen:

```wspl
untie parameter("n0.weight") as W_drive;
W_drive[0,0]=3;
```

Inside a graph-mode transaction with a host-supplied budget, `untie` gives that
consumer a private copy before the update. Other consumers retain the shared
weight. Region `replace`, `transplant`, `redirect` and `add_residual` operations
are also implemented; their boundary mappings and topology budgets are explicit.
See the [graph surgery guide](docs/wspl/WSP_GRAPH_SURGERY.md) for complete examples.

The editor interprets bounded construction syntax as data; it does not execute
arbitrary proposal Python. `plan` records the parent, candidate, write footprint
and rounding; `apply` replays the plan and checks identity. `describe` and `check`
attach separate evidence to frozen artifacts. Author-provided semantic labels
remain hypotheses. Formal checks cover the supported affine/ReLU real-graph
subset; editable tanh networks receive empirical execution checks, not proofs.
The [implemented editor guide](docs/wspl/WSP_EDITOR_FIRST_PASS.md) and
[study interface](docs/wspl/BREAKOUT_REWORKED_INTERFACE_V1.md) document the working
subset; the broader [syntax design](docs/wspl/WSP_SYNTAX_0_2.md) also includes
planned features. The historical Core PWA module frontend below is separate.

## Try the exact scalar reference

Python 3.11 or later. The reference core uses the standard library:

```sh
python -m pip install -e ".[dev]"
```

```python
import json
from pathlib import Path
from skillforge.wspl_core import ScalarProgram

program = ScalarProgram(json.loads(
    Path("examples/wspl/brake_patch.core.json").read_text(encoding="utf-8")))
result = program.evaluate({"d": "1/5", "v": "2/5", "a_base": "1"})
assert str(result["a"]) == "-1"
```

The same program can be written in the proposal's text form,
[brake_patch.wspl](examples/wspl/brake_patch.wspl), and parsed into this IR:

```python
from skillforge.wspl_parse import parse_module

parsed = parse_module(Path("examples/wspl/brake_patch.wspl").read_text(encoding="utf-8"))
assert str(ScalarProgram(parsed.program).evaluate({"d": "1/5", "v": "2/5", "a_base": "1"})["a"]) == "-1"
```

Constants and inputs are exact rational strings or integers; floating-point input
is not silently treated as an exact specification. The scalar IR supports affine
arithmetic, ReLU, clipping, normalization, min/max and graded Lukasiewicz logic.
Unit labels, bounds and acyclic references are checked. Conservative interval
analysis can reject a valid program when a stronger proof is needed.

See [the scalar core](docs/wspl/SCALAR_CORE.md),
[BrakePatch](examples/wspl/brake_patch.core.json) and
[ApproachP](examples/wspl/approach_p.core.json).

## Compile neural weights

```sh
mkdir work
python -m skillforge.wspl_compile examples/wspl/brake_patch.core.json --output work/brake_weights.json
```

Create the output directory first and choose a new file. BrakePatch compiles into
the proposed 3 -> 4 -> 2 -> 1 ReLU network, with 29 parameters. ApproachP compiles
into a 12 -> 6 -> 6 -> 4 network, with 148 parameters. Generated examples and the
independent loader are described in [the compiler notes](docs/wspl/COMPILER.md).
Optional Torch emission checks exact coefficient representation; floating-point
execution and universal contracts are not certified by compilation.

## Check contracts and investigate opaque networks

With `python -m pip install -e ".[dev,verify]"`, the initial verifier checks
serialized ReLU graphs using rational real arithmetic. The three BrakePatch
contracts and the approach bounds/direction contracts are solver-backed PROVED;
broken gain/input-order fixtures produce independently replayed counterexamples.
This does not establish source equivalence, IEEE execution or physical safety.
See [verification scope and examples](docs/wspl/VERIFICATION.md).

A separate check asks whether a compiled artifact computes exactly its scalar IR
source over the declared input box. Both committed examples are EQUIVALENT under
rational real semantics, and weight regressions return counterexamples replayed
through both interpreters:

```sh
mkdir -p work
python -m skillforge.wspl_equiv examples/wspl/brake_patch.core.json examples/wspl/brake_patch.relu.json --output work/brake_equivalence.json
```

This is not an IEEE execution proof or a proof-carrying certificate.

For networks outside that verifier, [intervention logging](docs/wspl/INTERVENTION_EVIDENCE.md)
records controlled activation replacements with baseline, no-op and repeat checks.
It supports tensor-valued PyTorch eval/stateless adapters, records identity/RNG
and output effects, and can replay/chain empirical evidence. Observed computational
effects are not semantic concepts, real-world causes or universal guarantees.

## Research program

The first proposed product workflow is **controller change review and repair**.
See the [customer validation plan and collaborator handoff](docs/PRODUCT_VALIDATION.md).
An inspected engineering demo compiles a controller, exposes a deliberate gain
regression with a replayed counterexample, and checks an operator correction:

```sh
python tools/demo_controller_review.py --output work/controller_review
```

Add `--include-opaque` in the tested Torch/NumPy environment to include paired
activation probes. This is a known-answer demonstration, not automatic repair,
customer validation or a scientific performance comparison.

The three studies cover fresh authoring, repair of independently trained artifacts
and frozen composition. They span discrete causal worlds, low-dimensional dynamics and
robot manipulation. Each compares direct weights, ordinary PWA code and WSPL,
with and without verifier feedback. All arms receive final verification.

The [experimental program](docs/wspl/EXPERIMENT_PROGRAM.md) separates mathematical
contracts, deployment numerics and physical outcomes. The proposed first screen
has 36 synthesis runs, but is not registered, funded, approved or launchable yet.
Astra is selected; spending authorization, sampling, fresh task freeze and isolated
collection controls remain pending.

The [formal experiment protocols](docs/wspl/FORMAL_EXPERIMENTS_V1.md) now specify
the first finite-domain authoring screen and subsequent learned repair/composition
studies. Implemented: six-arm trial kernel, common 541-parameter target, finite
verification, feedback firewall, budget/selection records, public request export
and source-gated offline grading. Isolated provider collection and frozen launch
settings for that six-arm screen remain pending. A separate native Breakout
authoring screen has completed, as reported below.

Initial claims are explicitly mapped to evidence in that protocol. Final selected
code/WSPL artifacts now receive a separate rational source-equivalence diagnostic;
Boolean task success alone does not establish source fidelity between test points.
The WSPL arm accepts Core PWA module text through the text parser; JSON IR remains
internal. Astra, target capacity and evaluation budgets stay matched across arms.

- [Structured study plan](experiments/plans/wspl_cross_domain_v1.json)
- [CSAIL comparison and product win conditions](docs/wspl/CSAIL_COMPARISON.md): useful performance, checked changes and less engineering/review effort; outperforming the published game score is optional
- [Completed Breakout artifact replay](docs/wspl/BREAKOUT_REPLAY_RESULTS.md): 511.828 mean return versus upstream 517.411; uniform control 0.551; 1,000 episodes each, CPU only, no model calls
- [Completed fresh Astra native authoring screen](docs/wspl/ASTRA_BREAKOUT_NATIVE_RESULTS.md): three one-shot policies, 3,000 evaluated episodes; all replies, usage and failed-attempt history archived
- [Earlier matched ReLU comparison](docs/wspl/BREAKOUT_MATCHED_V1.md): completed; interpretation paused during review of compiler/interface fairness
- [Reworked native/WSPL comparison](docs/wspl/BREAKOUT_REWORKED_V1_RESULTS.md): completed on the original tanh network; performance, token use and replayable artifacts summarized below
- [Implementation gates](docs/wspl/WSPL_IMPLEMENTATION.md)
- [Original supplied WSPL proposal](docs/wspl/PROPOSAL_v0_1_2026-11-03.md), archived unchanged with provenance
- [Related work](docs/RELATED_WORK.md); compilation and neural repair have substantial precedents

## Experiments

### Ordinary Python versus neural prediction programs

A follow-up added six ordinary Python predictors to the completed small-data
panel, reusing the existing native WSP, WSPL and AdamW results. All six were valid.
With the same 128 acquisition examples and two-response validation budget,
**Python achieved lower prediction RMSE than both neural authoring interfaces on
five of six tasks**, and beat both AdamW baselines on all six. ERCOT favored the
neural predictors; Dominick still favored a simple training-mean predictor.

Causal effects diverged from factual accuracy: IHDP effect RMSE favored Python
(0.375 versus native 1.654 and WSPL 2.945), while ACIC strongly favored native WSP
(0.0137 versus Python 0.0826 and WSPL 0.1222). Python used explicit nonlinear
formulas and example-based prediction, without an executable optimizer.

This exploratory, one-draw comparison weakens the case for neural deployment as
the default for standalone prediction. Python had greater expressivity; public
dataset familiarity, reused partitions and limited gradient baselines also limit
interpretation. WSPL's integration, composition and verification benefits need
separate evidence. No neural reruns were needed.

See the [full results](docs/wspl/PYTHON_ADDITION_RESULTS.md),
[audit](docs/wspl/PYTHON_ADDITION_RESULTS_AUDIT.json), and
[final programs](artifacts/python_addition_v1/).

### Small-data prediction panel: forecasting, regression and causal outcomes

The completed stage-one panel contains 24 problems (14 forecast series, eight
regressions and two causal instances), with one draw each of native WSP, WSPL,
Astra-guided AdamW and fixed AdamW. All 96 final candidates were valid. Methods
shared at most 128 acquisition examples and the same small ReLU architecture;
direct authors constructed weights without executable fitting. All candidates
were frozen before test scoring.

| Direct author | Lower factual RMSE than fixed AdamW | Lower factual RMSE than guided AdamW | Best of four |
| --- | ---: | ---: | ---: |
| Native WSP | 22/24 | 20/24 | 14/24 |
| WSPL | 24/24 | 23/24 | 9/24 |

WSPL used 342,036 output tokens versus native's 622,124 (including reasoning),
approximately 45% fewer, and approximately 43% less authoring time. Native won
14 of their 24 head-to-head accuracy comparisons. These are descriptive results,
not evidence of WSPL accuracy superiority or a compute-matched comparison.

Causal effects require separate evaluation: native's ACIC effect RMSE was
0.01368 versus WSPL's 0.12219, and WSPL estimated the wrong ATE sign despite
better factual prediction than either gradient arm. All neural methods also
lost to a training-mean predictor on three problems. One draw, small samples,
limited AdamW baselines and possible public-data pretraining exposure constrain
the interpretation.

See the [results and limitations](docs/wspl/PANEL_STAGE1_RESULTS.md) and
[compact audit](docs/wspl/PANEL_STAGE1_RESULTS_AUDIT.json). Full local evidence
is preserved; no automatic expansion follows this panel.

### Go: constructing neural policies without gameplay feedback

Six Astra/xhigh sessions programmed a fixed 1.48M-parameter ReLU policy from
zero weights: three using Python/NumPy, three using WSPL. Each had up to eight
response/construction steps, with construction diagnostics but no gameplay,
policy-output feedback or gradient training. All candidates were frozen before
42,000 CPU games against three same-size PPO opponents, including a zero control.

| Authoring draw | Python/NumPy win rate | WSPL win rate | WSPL minus Python |
| --- | ---: | ---: | ---: |
| 1 | 42.52% | 38.28% | -4.23 pp |
| 2 | 32.67% | 30.92% | -1.75 pp |
| 3 | 21.32% | **52.15%** | **+30.83 pp** |
| Mean | 32.17% | **40.45%** | **+8.28 pp** |

Rates average 6,000 games per policy over the three opponents. Zero control:
1.62%. WSPL's best policy exceeds the best Python policy by 9.63 percentage
points. The large third-pair win outweighs two smaller losses, but three authoring
replicates do not establish a reliable language advantage. These are opponent
win rates, not direct Python-versus-WSPL matches.

The strongest WSPL session repeatedly refactored explicit liberty, vulnerability,
connection and eye-handling computations inside the fixed network, using tensor
edits, loops, slices and parent references. It did not use graph growth or the
high-level circuit-placement syntax. Semantic correctness of its claimed
"certificates" and the contribution of individual edits remain unverified.
Python also produced explicit tactical circuits. WSPL used more reported input
and output tokens in this screen; there is no token-efficiency win here.

One Python request timed out before returning an answer. Recovery preserved four
frozen candidates byte-for-byte and completed the remaining pair; the failed
call's usage is unknown. All six candidates constructed successfully. This is a
smaller construction-budget screen than the original CSAIL experiment, not a
strict reproduction. See the [results and accounting](docs/wspl/GO_AUTHORING_V1_RESULTS.md),
[protocol](docs/wspl/GO_AUTHORING_V1.md), and
[recovery protocol](docs/wspl/GO_AUTHORING_RESUME_V1.md).

The [seven-edit review](artifacts/go_edit_sequence_v1/index.html) preserves exact
patches, source diffs, changed-weight lists and per-call usage/length metrics.
Paired evaluation peaked at edit 3 (54.13%) before declining to the final 52.15%;
all 189 final-checkpoint batches reproduced the earlier evaluation exactly.
This is post-hoc analysis on the original seeds, not held-out checkpoint selection.

### Diffusion: controlled coordinate editing

[Released-model replay](docs/wspl/DIFFUSION_REPLAY_V1_RESULTS.md) produced 3,072
images from three small ReLU denoisers using a declared DDPM reconstruction.
In the subsequent [Python versus WSPL comparison](docs/wspl/DIFFUSION_EDIT_V1_RESULTS.md),
all six fresh Astra programs correctly shifted generated images two pixels right
with wraparound. Each program transferred to all three models: 18/18 applications
passed, with bit-identical edited weights per parent and exact real-graph
permutation witnesses. This was a known coordinate transformation, not semantic
feature discovery or general graph surgery.

WSPL used 2,710 reported output tokens versus Python's 3,260 (16.9% fewer),
driven by reasoning tokens; non-reasoning output was slightly larger. Local
construction took 54.8s versus 9.1s. Both arms reached the task ceiling, so this
establishes feasibility rather than an accuracy or general efficiency advantage.
WSPL used loops and tensor copies, not circuit placement. The complete screen
finished in 4.04 minutes. [Raw programs and replies](artifacts/diffusion_edit_v1/index.json)
and [audit](docs/wspl/diffusion_edit_v1_audit.json) are retained.


### Kinetix: repair of a supplied trained controller

Six fresh Astra/xhigh proposals edited the same 171,664-parameter FP32 parent,
with identical task information, fixed topology and 1,024 paired episodes each.
The parent scored 0/1,024; **every native and WSPL repair scored 1,024/1,024**.
The descriptive feasibility rule passed, with no task-performance advantage.

WSPL used 29,706 output tokens versus 38,315 native (**22.5% fewer**, including
reasoning). That aggregate difference is largely driven by one native draw;
median output use differs by only 3.3%, and input+output totals by 0.78% because
the supplied weights dominate context. All WSPL programs chose the
native-compatible literal profile.

Across three proposals, WSPL used **97 edit statements versus 112** and
**1,949,335 charged construction operations versus 2,487,951** (21.6% fewer).
These counters measure editor work, including tensor processing, not inference
FLOPs; both arms still execute the same dense architecture. The larger third
native repair drives much of this aggregate difference. Five of the six fresh
repairs used 24–36 statements, compared with 51 in the published repair; the
published float64 replay is historical context, not a matched control here.

The protocol deliberately excluded circuit placement and graph surgery to hold
topology fixed. Constants, exact arithmetic and loops were available but unused.
Literal edits sufficed for these compact run-up controllers; presenting the
compatibility profile first may also have influenced selection, but that is an
untested explanation. This establishes useful repair through either interface,
not an advantage from richer constructs. The next discriminating comparison
should allow structured surgery alongside equivalent raw graph access and test
preservation obligations or varied conditions.
[Results, limitations and audits](docs/wspl/KINETIX_REPAIR_V1_RESULTS.md);
[all replies and source programs](artifacts/wsp_kinetix_v1/README.md).

### Matched native versus WSPL: performance and token efficiency

Six fresh Astra/xhigh proposals used the same task information, zero-initialized
20,486-parameter tanh network and 1,000 paired evaluation episodes per policy.
Three proposals used native literal weight edits and three used the rebuilt
WSPL interface. All six were valid. No author received tools, evaluation feedback
or gradient training.

| Measure | Native | WSPL | Observed change with WSPL |
| --- | ---: | ---: | ---: |
| Mean reward across three proposals | 370.516 | 391.685 | +5.7% |
| Best proposal reward | 498.366 | 524.558 | +5.3% |
| Total output tokens, including reasoning | 66,085 | 53,167 | 19.5% fewer |
| Reasoning output tokens | 55,737 | 49,137 | 11.8% fewer |
| Other output tokens | 10,348 | 4,030 | 61.1% fewer |
| Stored executable source bytes, all three | 16,185 | 7,131 | 55.9% fewer |

Token and byte totals cover all three proposals per arm; reasoning and other
output tokens partition total output. Input totals were closely matched:
12,177 native versus 12,186 WSPL. Native stored sources include their post-hoc
WSPL normalization wrapper, which was never supplied to either writer.

All three WSPL programs used construction loops and exact arithmetic. Repeated
weight patterns could be expressed once and expanded by the editor. The largest
token reduction was in non-reasoning output, consistent with more compact weight
construction. These observations motivate a larger efficiency study; they do
not establish why reasoning usage changed or a general cost advantage.

For the best pair in this screen, WSPL used **2,864 source bytes versus 5,913**
(51.6% fewer) and **17,504 total output tokens versus 19,308** (9.3% fewer),
while scoring 524.558 versus 498.366. Our best earlier native policy scored
**541.818**, above the best WSPL score here; that earlier prompt package and
evaluation seeds differ, so it is historical context rather than a matched control.

Across-proposal reward SDs were 120.0 native and 125.2 WSPL. **Three authoring
draws per arm do not establish performance superiority or a reliable efficiency
advantage.** The predeclared descriptive progression criterion passed. No formal
tanh/controller proof is claimed, and shorter source is not itself evidence of
easier human review.

All replies, sources, weights, edit plans, equations, episode records and the
first three trajectories per policy were retained. The audit reconstructed all
six candidates, checked all 6,000 episode records and replayed 127,378 recorded
actions from their observations and RNG states.

- [Full results and limitations](docs/wspl/BREAKOUT_REWORKED_V1_RESULTS.md)
- [Best matched WSPL program](artifacts/wsp_breakout_v1/draw_0_wspl.source.wspl)
- [Best matched native program](artifacts/wsp_breakout_v1/draw_0_native.answer.md)
- [Best historical native program](artifacts/astra_breakout_native_v2/draw_1.answer.md)
- [All six replies and executable sources](artifacts/wsp_breakout_v1/README.md)

### First fresh native weight-authoring screen: Breakout

In a supplied-rule pilot, **Astra authored functioning neural Breakout controllers
in one response per draw**, using the original CSAIL compact weight-edit syntax.
Each starts from a zero-initialized 600 -> 32 -> 32 -> 6 tanh network with 20,486
parameters. Three separate fresh conversations received the same game rules and
source, with no reference solution, evaluation feedback or gradient updates.

| Fresh policy | Episodes | Mean return | Episode standard error |
| --- | ---: | ---: | ---: |
| Draw 1 | 1,000 | 484.756 | 4.469 |
| Draw 2 | 1,000 | 541.818 | 5.000 |
| Draw 3 | 1,000 | 510.042 | 7.755 |

The three-policy mean is **512.205**. Our separate released-policy replay scored
511.828 and its uniform control 0.551; those used different evaluation seeds and
provide context rather than a paired superiority test. Every fresh policy used
the same 1,000 evaluation seed pairs. All 3,000 episodes, including horizon
truncations, are retained. All edits parsed successfully, and the event logs show
one completed turn and zero tool calls per draw.

The completed run took 41.1 minutes, including 29.7 minutes collecting replies.
Reported usage: 10,020 input tokens (3,200 cached) and 65,775 output tokens,
including 55,638 reasoning tokens. An earlier invocation aborted on a CLI startup
warning is also recorded; its usage is unknown. The read-only audit checks source
identity, raw-file hashes, reply-to-weight reconstruction and every episode record.

This demonstrates useful native weight authoring on one supplied task. The later
matched WSPL comparison is reported above; a gradient-trained control and formal
controller proof remain future work. See the [full results and limitations](docs/wspl/ASTRA_BREAKOUT_NATIVE_RESULTS.md),
[machine-readable audit](docs/wspl/astra_native_v2_results.json) and
[all three authored replies](artifacts/astra_breakout_native_v2/README.md).

## Path to Recursive Twin Improvement

**Recursive Twin Improvement (RTI)** is a proposed learning loop in which two
frontier-model instantiations take turns editing each other's weights. Each edit
must demonstrate independently evaluated improvement in the target domain while
satisfying required contracts and preservation criteria before it is accepted.

WSPL is intended to make these edits explicit, executable and checkable. Accepted
and rejected proposals, counterexamples, artifact identities and evaluation
outcomes would form training data and rewards for models to become better weight
editors. This couples two objectives: improving the target's domain capability
and improving the editor's ability to produce useful weight changes.

The proposed twin cycle is:

1. Model A proposes a weight edit to model B.
2. An independent evaluator checks the edited artifact, domain improvement and
   preservation of required behavior. Accept the edit only if those criteria pass;
   otherwise retain the previous target version and record the failed proposal.
3. Model B proposes an edit to model A under the same evaluation rules.
4. Alternate roles, retaining the edit histories and using their outcomes to
   train and test subsequent editing ability.

The evaluator, requirements and protected evaluation data remain outside the
twins' editable state. Mathematical contract results and empirical domain scores
are recorded separately: a measured improvement is not a universal guarantee,
and opaque-network intervention logs are not proofs. The current verifier covers
small supported graphs, not arbitrary frontier-model edits.

An accessible precursor uses frontier models as editors of smaller networks with
editable weights. Genuine frontier twins additionally require editable frontier
weights. The planned authoring and learned-repair studies would test prerequisites;
RTI, reward-based editor training and recursive capability gains are not implemented
or demonstrated here.

The decisive experiment must show that later versions become better **editors**,
as well as better domain models. Compare the alternating loop with a fixed editor,
self-editing and independent training at matched compute, using fresh evaluation
tasks. Successful alternating edits alone do not establish recursive improvement.

## Existing robot evidence

A small, unblinded engineering pilot compared six controllers on positioning at
a supplied standoff above an object. Each neural block had about 1,600 parameters.

| Controller | Successful evaluation starts |
| --- | ---: |
| Zero translation | 0/8 |
| Reference proportional controller | 8/8 |
| Analytic MLP weights encoding the reference | 8/8 |
| Analytic RMV weights encoding the reference | 8/8 |
| AdamW MLP, four demonstrations | 5/8 |
| AdamW RMV, four demonstrations | 3/8 |

The complete attempt took 56.266 recorded CPU seconds. RMV/scalar discrepancies
on recorded visited inputs were below 3e-8. This is empirical numerical agreement,
not a universal floating-point guarantee. Earlier failed attempts and costs are
retained in the [audited report](docs/APPROACH_V1_RESULTS.md).

These were inspected analytic encodings of a known controller, not a blinded LLM
synthesis result. The task was pregrasp positioning, not grasping, pickup, general
transfer or physical hierarchy execution. Four demonstrations did not reliably
teach the same closed-loop behavior; a general LLM-over-gradient advantage is not
established.

## Robotics testbed

The selected stack is MuJoCo for physics, robosuite for robot/control infrastructure,
and LIBERO for tasks and assets. The local adapter was checked against MuJoCo
3.2.7, robosuite 1.4.1 and a pinned LIBERO checkout. Simulation is optional and
must be installed/configured separately; reference tests do not launch it.

- [System specification](docs/SYSTEM_SPEC.md) and [adapter contracts](docs/SIM_ADAPTER.md)
- [Sixteen robot task specifications](docs/ROBOT_TASK_SUITE.md): repeated grasps,
  sorting, drawer/door chains, barriers, blocker clearance, timed holding and recovery
- [Symbolic hierarchy fixtures](docs/HIERARCHICAL_TASKS.md)
- [LTLf compilation and strong planning](docs/M1_ENGINEERING.md)

Obstacle/articulation/orientation inputs, contact-ready skills, independent
physical observers and the primitive executor remain explicit integration work.
An existing BDDL scene is not a working Skillforge task.

## Checks and reproducibility

```sh
python -m unittest discover -s tests -v
```

The suite now contains 426 checks: 425 pass in the primary research environment,
with the native MinAtar fixture skipped there. All 12 replay fixture checks pass
in the separate MinAtar CPU environment, including that fixture. Fifty editor
checks cover tensor compatibility, placement, transactions and evidence; fifty
graph checks cover cut boundaries, rewiring, shared storage, budgets, rollback,
CLI/examples and edited-graph claims. The [validation receipt](docs/wspl/wsp_graph_surgery_validation.json)
records the graph milestone; [repair preparation](docs/wspl/wsp_repair_preparation_validation.json)
adds 18 harness checks and current source identities. Optional neural/adapter checks
skip when PyTorch or NumPy is absent, and solver checks skip without Z3; full local results used a separate CPU
research environment. No model download, simulator rollout, scientific training
run or LLM API call occurs in the unit suite. A single optimizer transaction is
used only as a checkpoint/gradient unit fixture.

[Current validation receipt](docs/wspl/wsp_editor_first_pass_validation.json) and
[test output](docs/wspl/wsp_editor_first_pass_unit_tests.txt) record the checked
sources; earlier receipts remain historical records. Scientific runs require an exact user-approved source
manifest through [the source gate](experiments/source_pins/README.md). Historical
pins are preserved, not refreshed to match later development. Model weights,
datasets, credentials and machine-specific bindings are excluded from Git.

The [archive check](docs/github_archive_check.json) also records 132 passing tests
from a fresh GitHub clone, using the existing external CPU dependencies.

Install simulator environments **outside this checkout** for source-gated runs;
the local-source gate rejects unpinned package code inside the repository. Use
ignored `artifacts/local_paths.json` for machine bindings.

## Project record

- [Working context](docs/CONTEXT.md)
- [Evidence and limitations](docs/EVIDENCE.md)
- [Research roadmap](docs/ROADMAP.md)
- [Provenance](provenance/README.md)
- [Research coordination contract](AGENTS.md)

This repository preserves code, proposals, results and negative findings for the
ongoing research program. It is separate from legacy Gravemind, RBL-RMV and TNST
projects; their artifacts are not silently relabelled as WSPL results.

GitHub: https://github.com/k-oconnor/WSPL (private research archive).
No redistribution license has been selected. Nothing in the proposed experiment
program authorizes new API spending, training, deployment or physical hardware use.
