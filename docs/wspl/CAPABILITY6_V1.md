# 6x6 capability engineering screen

Prepared experiment, not launched. User accepted6x6 and requested proceeding.
One source-approved screen will compare unchanged foundation, fixed gradient
training, two native Astra authors and two native-plus-WSPL Astra authors.
Order:gradient,native0,WSPL0,WSPL1,native1. Each starts independently from the same
copy-trained SmolLM2-135M foundation and same zero-write rank16 attention branches.
No previous movement adapter is restored. Base weights always frozen.

Default120q/k/v/o attention branches across30layers,1,891,200 trainable parameters.
Common author envelope2.5M parameters,120sites,16nodes/depth8/250kparameters per
static vector DAG. All native operations remain available in WSPL arms. A shared
host bridge maps finite FP32 DAG storage to trainable Torch parameters and adds
outputs to named projections. Explicit sharing retained within a branch; no
cross-branch tensor sharing. It is not a whole-transformer WSPL compiler.
No semantic/general IEEE proof is claimed from branch execution checks.

## Data and representation

Seed6101,6x6,background0,object1..8,marker9. Connected free polyomino families
canonicalized under all rotations/reflections. Training uses all1/2/3cell families
and first3lexicographic tetromino families. Novel families:remaining2tetrominoes
and all12pentominoes. Larger transfer:all35hexominoes. Generation excludes
collisions/out-of-bounds. Whole inputs disjoint across every split and task;
4x4copy fixtures from earlier foundation preparation are excluded as well.
Sampling families/orientations/placements/markers/colors is deterministic, but
does not promise perfectly balanced final counts; archive actual distributions.

Training16,384rows:8,192movement+4,096copy6+4,096copy4. Development1,024rows:
512movement+256copy6+256copy4. Movement rows are6x6;4x4copy uses0..3 digits.
Six final panels:familiar512,novel512,larger512,copy4 512,copy6 512,composition256.
Final predictions start only after all five trained artifacts freeze. All old
results are development background; no old final grid is reused.

Fixed instruction-only prompt,zero worked demonstrations,noCoT. This differs
from the historical eight-demo prompt, so compare only the same-prompt foundation
baseline in this study. Greedy unrestricted16/36-token output, native KV cache,
no extraction/repair/retry. Single-digit tokenizer mapping checked. Cache/no-cache
generation parity checked synthetically. Two-call composition passes generated
grid cells unchanged into the same prompt/model; no movement oracle runs between
calls. Both intermediate and final outputs must be correct. Invalid first output
fails without a second call. Test generator guarantees both true moves are valid.

## Common budgets and authors

Each trained arm:at most8,192updates or3,600seconds of synchronized training-step
wall time, whichever comes first. Batch4,FP32,AdamW,zero decay,clip1. DefaultLR.001,
32updatewarmup each chunk. Four chunks of2,048 for gradientcontrol; up to four
responses/chunks for each Astra/xhigh author (16calls maximum). Optimizer moments
reset per chunk for all arms. Construction,export and evaluation time reported
outside training allowance; one update may cross the time bound. All attempts
consume the same per-arm allowance; rejected/FINAL responses can leave it unused.

First512cumulativeupdates arecopy-only,equalcopy4/copy6. Subsequent defaultmix
75%movement/12.5%copy4/12.5%copy6. Both author groups can choose placement, rank,
supported topology and initialization plus LR/copy fraction/changed-cell loss
weight within declared limits. This is a bounded shared training service, not
unrestricted Python/PyTorch author execution or arbitrary training-code search.
Dataset labels available through the host; author packet exposes the same fixed
eight examples per train/dev pool, aggregate exact-development scores and first
eight output records per pool. No reserved feedback. No activation/gradient API
in this screen. Last accepted trained artifact is final; no best-checkpoint search.

Five training arms imply at most five synchronized training hours. Synthetic
default update~.27s suggests~36minutes per full8,192updates, not a runtime guarantee.
Development evaluation occurs after every accepted chunk, including control;
up to20,480development grids. Final6conditions x2,816rows=16,896scored rows,
up to18,432model generations including second composition calls. Evaluation and
authoring add materially to training time. Estimate5–10hours,18hourstudyguard,
20minuteinvocationguard. ExistingCodexusage,noharddollar/outputtoken cap or API
fallback. No automatic retries or follow-on. Unexpected/nonfinite training aborts
the study with incomplete status; never silently substitute a successful run.

## Independent criteria and interpretation

Per trained arm, require all:
- Familiar487/512,novel461/512,larger461/512 exact or better.
- At least103additional exact successes over same-prompt unchanged foundation
  in EACH movement panel (about20percentage points).
- Both copying panels507/512 or better and at mostfive paired regressions each.
- Composition231/256 or better.

These are engineering progression thresholds, not power-derived significance.
If baseline already solves the task, the gain criterion prevents claiming new
capability. Even a pass establishes only a bounded synthetic neural skill.
Compare authors to ordinary-gradient control to assess authoring value. Compare
native versus native-plus-WSPL descriptively with two sessions each, reporting
actual WSPL use, failures,tokens,time,trainingtargets and retained skill. Language
availability alone is not language advantage; all deployable expressivity is shared.

Archive source identity, prompts,replies,construction transactions, detailed dev
and final predictions, stage budgets and all final module artifacts. Each arm has
one rolling resumable tensor/optimizer/RNG checkpoint every30minutes and at chunk
end/clean-stop, overwritten thereafter; no best/per-improvement training checkpoints.
Separate author construction artifacts are retained for replay/audit (potentially
hundreds of MiB), and one final deployable module plan per arm. Parent full model
is referenced, not copied. No changes to historical source-pinned runs.
