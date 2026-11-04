# Matched Breakout ReLU interface comparison

Prepared protocol, November 3, 2026. No matched-screen proposer call, game scoring
or source registration has occurred. Launch requires a new exact-source approval.

## Question and frozen design

Does WSPL change fresh one-shot policy authoring performance/cost relative to
competent native weight edits on the same fixed target? A restricted Python PWA
body control separates symbolic source construction from WSPL typed syntax.
This is an exploratory three-interface screen, not confirmation of superiority.

Three arms, three fresh conversations each: nine proposals. Same Astra/xhigh,
ChatGPT-authenticated restricted Codex collector, task/rules/source information,
raw 600 inputs, six sampled actions, fixed 600 -> 32 -> 32 -> 6 ReLU network,
20,486 FP32 parameters, one reply, no tools, gradients or score/proof feedback.
No prior authored policy, reply or measured score is supplied to the proposer.
Existing tanh results are historical context, not controls in this comparison.
The move from tanh to ReLU is a labelled extension of the task, not replication.

All arms receive byte-identical common prompt content: original game rules/source
and observation/action mapping plus the entire shared three-interface packet.
Only a final selected-arm line differs (at most three UTF-8 bytes). Source arms
get no hand-coded spatial features, extracted coordinates, planner or policy
adapter. Every accepted policy runs the same NumPy ReLU tensor implementation.
The exact prompts and the generated bundle are bound in the source manifest.

Order rotates to balance serial position:

| Block/draw | First | Second | Third |
| --- | --- | --- | --- |
| 0 | weights | code | WSPL |
| 1 | code | WSPL | weights |
| 2 | WSPL | weights | code |

Blocking does not pair the provider's unexposed random seeds. Fresh threads are
checked; prefix-cache usage is recorded and is not shared conversation history.
Provider sampling/seed details beyond the CLI configuration are uncontrolled.
No best-of selection, retries by the harness, repair or automatic follow-on.
Bounded CLI transport retries may occur, as in the completed native screen.

## Representations, allocation and numerical scope

[The shared packet](BREAKOUT_MATCHED_INTERFACE_V1.md) is the exact public guide.
Weights retain the native competent tensor slice/broadcast/update language and
its logged invalid-statement skipping. A complete zero-weight proposal is scored.
Code uses fresh scalar assignments and a final six-value return in ordinary
Python syntax. It is AST-parsed as data, not executed or arbitrary Python scripts.
WSPL uses existing Core PWA text with typed lets and six output assignments; the
harness inserts the common fixed port declarations. This snippet profile is
explicit, not general vector/array WSPL or full module authorship.

Both source arms lower to the same scalar IR/compiler. Expanded WSPL, IR and
deployment artifacts are saved. Intervals are inferred for untyped code; WSPL
declared intervals must satisfy the existing type checker. Neither gets a
reference controller or a compilation search. Unsupported source/depth/width
fails rather than silently increasing capacity. All hidden relays and zeros
count in the exact fixed target. Pure affine outputs use signed output relays;
nonnegative hidden states use identity relays. No extra skip connections.

The original 1,024-node source limit would arbitrarily constrain a 600-input
target. This profile allows 50,000 scalar IR nodes while preserving the old
default for other callers. Python AST <=200,000 nodes; whole replies <=1 MB.
Expression depth stays <=32; WSPL parsing retains its own <=64 nesting and1 MB
expanded-module limits. Compilation guard <=128 source/relay units, <=100,000
temporary dense parameters and <=2 nonlinear layers. Final hidden widths must
both be <=32 before padding to the common target. These are implementation
limits, not impossibility certificates or general-language expressivity claims.

All final coefficients have magnitude <=1e6 and are FP32. Native edits use FP32
updates. Exact compiled source coefficients are explicitly converted via FP64 to
FP32; changed-coefficient counts/max error are reported. The deployed FP32 values
are serialized as exact rationals in the final graph artifact. A final-only
5-second Z3 equivalence query compares source IR against that actual rounded
graph on the full [0,1]^600 box (20,486-parameter /50,000-source-node budget).
EQUIVALENT, DIFFERENT, UNKNOWN and INVALID are separate outcomes. Rounding can
legitimately yield DIFFERENT. Both source arms get the same diagnostic; weights
report NOT_APPLICABLE, since no independent scalar source exists. This is a
secondary rational-real fidelity diagnostic, not a task-success gate, IEEE proof
or proof of winning/safe game trajectories. Parser/compiler/solver trust remains.

## Evaluation and predeclared interpretation

Every accepted policy gets the same 1,000 environment/action seed pairs,
710000+i /810000+i (i=0..999), six-action sampled FP64 softmax, sticky actions0.1,
difficulty ramping and a10,000-step horizon. These seed pairs differ from both
earlier screens. All complete episodes, including horizon-truncated returns, are
retained. Partial stopped episodes are logged/excluded. Local CPU/one math thread,
no local GPU or optimizer. The hosted model's compute is not locally measured.

Primary descriptive endpoint: mean of three proposal-level scores per arm.
An invalid proposal contributes zero to the explicitly named failure-adjusted
score; zero is an operational failure penalty, not a measured game return.
Also report valid-only performance, valid counts, per-policy episode SE and
between-policy variability. Collection/access/guard failures stop the queue;
unattempted/partial trials remain visible. Incomplete comparisons cannot yield
a go decision. Invalid source proposals do not trigger replacement calls.

Prospective internal feasibility rule: both baseline arm means >=100; WSPL
three-of-three evaluable proposals, mean>=100 and >=90% of each baseline's
failure-adjusted mean. If baselines are not useful, comparison qualification is
unresolved. Otherwise failure yields no-go for this configuration. This fixed
100-point usefulness floor and90% tolerance are pilot decision choices, not
customer requirements or confirmed noninferiority. Native is the primary
representation control; the restricted code arm is supplementary. No assurance
or workflow-time claim is qualified by this performance-only rule.

Secondary: source fidelity, rounding/size/complexity, reported input/output and
reasoning usage, authoring latency, construction/verification/evaluation time,
parse failures and tensor identities. Record human preparation/review effort as
unmeasured; do not substitute token savings for verified engineering savings.
Three draws on one known supplied task cannot establish broad superiority,
transfer, learned repair, RTI or commercial demand. Public game/release exposure
in model pretraining cannot be excluded; all new replies remain in the ledger.

## Resource and execution boundary

Nine additional fresh Codex invocations; <=1,200 seconds each, whole screen
<=14,400 seconds (four hours). Expected approximately two hours, uncertain.
Existing ChatGPT usage is consumed; no enforceable dollar/output-token cap and
no direct API-key fallback. The earlier $150 suggestion is not spend approval.
No additional credits are provisioned. Use the existing audited restricted CLI
configuration and preserve all raw events/answers/commands/usage; the residual
question tool is still advertised and any tool event invalidates collection.
No live model calls are used in preparation or unit testing.

A new isolated venv-wspl-match pins the prior NumPy/MinAtar versions plus Z3;
previous run environments remain untouched. Scientific identity includes core
dependency code/binaries plus solver code/DLL hashes, not just package versions.
One final binary policy per evaluated proposal, exact deployment graph and raw
source/reply evidence; no best/periodic checkpoint proliferation. Source-gate
receipt and all bound prompts/configuration/dependencies must verify at launch.
Use isolated Skillforge-breakout-matched-v1; no source edits during execution.
Unit/fake-collector/short wiring checks are engineering checks, not scored trials.
