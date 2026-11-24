# Broader-access matched robot authoring

Prepared November 24, 2026. Not registered, approved or launched. Successor to
ROBOT_AUTHORING_V2.md; a new exploratory screen, not a continuation of its authors.

## Purpose and unchanged deployment

The previous authoring interface permitted196 LoRA projection pairs but supplied
only final-down-projection diagnostics. All six authors edited that pair. This
screen removes the privileged feature pack and offers chosen-layer diagnostics
and full-model acquisition replay equally to native Python and Python-plus-WSPL
authors. It does not require authors to edit several layers; coordinated edits
are supported and authors choose the sites.

Same frozen Qwen2.5-Coder-1.5B-Instruct base, original ordinary-practice parent,
196 rank16/alpha32 LoRA pairs, full-vocabulary one-token greedy action selection,
raw robot codec, supplied analytic motor blocks and physical mission semantics.
Only factor arrays deploy. Inspection services, author Python and author access
are absent during robot execution. No new architecture, runtime solver or prompt
change. Shared motors already use WSPL in all conditions; native refers only to
the authoring interface.

## Information and query interface

All authors receive original factors, complete projection catalog, the255 saved
acquisition decisions with teacher actions and tokenized prompts, raw codec and
parent development summary. The copied legacy features.npz is removed before
authors receive inputs; diagnostic_projection/feature_shapes are removed from
the catalog. The prompt no longer singles out a final-layer submission example.
Original archived artifacts remain unchanged. No previous author's transcript,
candidate, score or final result is supplied.

Each author can make at most8 acquisition inspection attempts. Each request has
exact keys tool,candidate,rows,projections,tokens,weights. candidate=null uses the
original parent; otherwise it is a relative native/WSPL submission path. The same
trusted reconstruction boundary used for development evaluates its shape, finite
values, original identity and construction. Inspection never makes that artifact
the deployed candidate; a separate evaluate call is required.

Rows must be original acquisition indices, or all255. No arbitrary prompts,
external paths, author-provided labels, code, development/final data or gradients
are accepted. Authors can choose zero to4 projection names from all196. Zero
names requests full-model replay/scoring. Selected projections return actual
input/output activations for the last token or all tokens. Full-sequence requests
are limited to32 rows. Optional frozen projection weight/bias matrices are
provided as FP32 copies. All queries return final residuals, normalization scale,
nine action-head rows/logits, full-vocabulary greedy tokens/top5 logits and exact
teacher-target NLL. The entire candidate runs through the actual frozen model,
including downstream nonlinearities and interactions.

Each query allows at most64M output elements (about256MB numeric payload) and
300seconds checked between forwards and before output save. Authors can reduce
rows/projections or omit weight matrices to fit this bound. Read-only NPZ/JSON
outputs are copied to a new query_NN input folder; the author sandbox cannot alter
them. Every output records the adapter identity and selected row/projection
mapping. Host parameters are restored and hooks removed even after a failed
forward. Frozen-base identity is checked after each author and at completion.

The full base checkpoint is not mounted in the sandbox. Chosen frozen matrices
and full-model acquisition forwards are exposed through this shared service.
All other Python numerical libraries and sandbox restrictions are unchanged.
Analytic supervised fitting is allowed; gradient training/iterative optimization
remains outside the primary scope. Source audit must disclose violations rather
than rebrand them. Whole-matrix WSPL assignments remain permissible but are
classified as serialization, not synthesis.

## Arms, reuse, data and final boundary

Six fresh Astra/xhigh author sessions, order native0,wspl0,wspl1,native1,native2,
wspl2. Independent sandboxes, no cross-author access,24 responses per author,
including malformed responses/finish;8 inspection attempts and4 development
evaluation attempts, including invalid requests. Inspection requests consume
responses, but do not consume development attempts. Both interfaces receive
identical information and numerical services. No native CLI tools are exposed.

Reuse the verified original255 acquisition decisions,16 parent development
episodes and frozen384-update gradient control from the failed v1 prefix. No
new teacher collection, control training or privileged static diagnostic pass.
This reuse is a new screen, not a claim that earlier computation cost was zero.
Historical expenditure remains recorded in the original studies.

Acquisition panel73300000 and development73400000 remain unchanged. Development
is a reused exploratory panel. NEW final panel73600000,16 missions per condition,
64 per frozen arm; previous final73500000 is excluded. Per-condition seed offsets
and3/4/5 waypoint counts follow the existing generator. Both controls (original
parent, reused gradient control) and all six author artifacts freeze in the ledger
before any final simulator opens. Maximum512 final missions. Invalid authors
remain failures; there is no fallback adapter. Last accepted development proposal
freezes even if its score worsens. Final outcomes never select an adapter.

This samples the same engineered domain. No broad language-retention, hardware
transfer, general calibration-algorithm or WSPL-superiority claim follows.
There is no contemporary narrow-access author arm: historical score differences
cannot establish that broader access caused an improvement, especially with a
different final panel and larger response allowance.

## Budgets, outcomes and stopping

At most144 fresh hosted author requests (24x6),48 acquisition query attempts
(8x6),24 development attempts (4x6). At most12,240 acquisition example-forwards
if every query uses all255 rows. Existing Codex usage, no hard dollar/token cap.
Twenty minutes per author response,300seconds per Python call,180seconds per
reconstruction. Existing AppContainer16-process/16GiB committed-memory limits;
shared GPU, no separate VRAM quota. Development wall24hours, measured new
development execution8hours, final6hours; outer watchdog30hours+5minutes.
Author waiting counts toward wall time. Query reconstruction/execution/copy time
counts toward development execution. Guards between bounded operations may
observe a limit just after the current operation. Allow roughly3-8hours; this is
an estimate, not a completion promise. Maximal query output may consume tens of
GB including the archived and sandbox copies.

Same predeclared exploratory criterion: parent ordinary>=12/16, candidate
ordinary>=14/16 with<=1 ordinary regression,>=8 net gains over48 adaptation
missions and no condition-level net loss. Report all paired counts, ordinary and
adaptation separately, invalid tokens, modified sites, query/response counts,
construction classification and time/token usage. Three author draws per
interface. The gradient control edits all196 pairs and remains descriptive,
not compute- or feedback-matched. Inspect whether broader sites were actually
used; eligibility alone is not evidence of their use or value.

Transport/simulator/infrastructure failure stops the study. No automatic retries,
follow-on, training or final-score-driven repair. Exact source-manifest approval
and separate registration are required before launch. This document authorizes
neither hosted scientific calls nor scientific forwards.

## Engineering evidence

New tests cover multi-site/full-sequence diagnostics, selected frozen matrices,
exact full-vocabulary NLL, invalid row/name/schema rejection, capacity limits,
error-path factor/hook restoration, inspection caps independent of development,
neutral prompts/inputs, fresh final seeds and all-authors-frozen-before-final.
Existing collector, full adapter reconstruction, source gate and robot checks
remain in the focused suite.

Full-size synthetic Qwen fixture changes early attention and middle MLP pairs,
checks native/WSPL factor and diagnostic bitwise equality, verifies changed
full-model output and exact original restoration/frozen-base identity, preserves
pre-existing model hooks, and confirms the real AppContainer can read query
outputs but cannot write them. Synthetic arbitrary token inputs only; no new
research predictions, author calls or training. Both actual offline author prompts
are rendered and checked with zero exposed CLI tools.
