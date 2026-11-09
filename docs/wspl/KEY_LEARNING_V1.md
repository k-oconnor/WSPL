# Learning-curve study: regularization selection with development reward

Prepared November5,2026; exact source approval needed. Assumed scope is expanded
heat-transfer examples for the fixed conduction value, not all natural ARC tasks.
This tests intervention generalization within one concept; cross-concept transfer
of the adaptation procedure is a separate future experiment.

## Data and limits

Designer-defined factorial questions:1024training,128development,256final. Four
balanced mechanisms:conduction,convection,radiation,evaporation. Split-disjoint
locations/object combinations AND template IDs. Training has32settings and8
templates per mechanism; development16settings and2templates; final32settings
and2different templates. No exact duplicates across splits. Water/ethanol and
elementary physical processes are supplied; labels follow construction.

This is templated, correlated data, not1024independent science problems or an ARC
benchmark expansion. The learning curve could reflect vocabulary/template coverage
rather than reasoning acquisition. Report per-setting/per-mechanism performance.
The earlier13/32case studies informed design; none is relabelled as new held-out
evidence. The value and initial broad key come from the first attention pilot.
Every fit shares that prior target-derived information. Holdouts prevent exact
context/template reuse but cannot prove separation from frontier pretraining.

## Inner learning and outer selection

Frozen base/value/slot; optimize ONLY64keycoefficients, using cached queries and
native log partitions.1000full-batch CPU FP64 Adam steps,LR0.03,clip1,projection to
the existing norm bound,FP32export. The same original broad key and fresh optimizer
initialize EVERY fit. Exact objective and8bounded recipe fields in
KEY_LEARNING_INTERFACE_V1.md. This is explicitly gradient-trained adaptation.
Preservation is encouraged by a negative-attention proxy and measured on actual
answers; the learner does not differentiate through full-model answer loss.

Compare two recipe-selection procedures, four development trials each at n256,
subset seed17, same updates/data/full-model development feedback:
1. Predeclared four-recipe grid:default;stronger negative/retention penalties;
   stronger ridge plus head-balance;combination. Exact values in runner GRID.
2. Astra/xhigh proposes four bounded JSON recipes adaptively after development
   feedback. No arbitrary code, other tools or optimizer changes. Invalid replies
   remain charged. Grid outcomes are not exposed to Astra.

Both select highest fixed development reward, earliest tie. Reward equals positive
accuracy gain minus2times the fraction of initially correct negative answers lost,
minus0.5times negative mean attention. Components/denominators always reported;
zero retention denominator makes that component uninformative. No valid Astra
recipe uses the predeclared default fallback. This is matched maximum trial/fit
budget, not matched prior knowledge, hosted compute or adaptive search strategy.
Astra receives feedback in its visible transcript; its own weights are not trained.
This is bounded adaptive recipe selection, not a differentiable bilevel optimizer.

Freeze selected recipes BEFORE fitting curves. n16/64/256 with three balanced,
nested subset seeds17/29/43; n1024one fit per recipe (all data, no fake replicas).
Twenty curve fits total; eight recipe-selection fits maximum;28,000optimizer
updates maximum. Fixed update counts imply more example exposures at larger n,
not compute-matched scaling. Subset seeds are data variation, not independent
author/optimization replications. Preserve every key/recipe/subset/trace.

## Evaluation and safeguards

Load/score final cases only after both recipes AND all20keys freeze. Compare each
curve key with untouched,original broad key,forced-value controls. Report scores,
attention selectivity/AUROC, corrected/broken cases, and retention denominators
by mechanism/setting. No post-final threshold or recipe tuning. Results inform
the next experiment, not a general claim that Astra beats optimization.

All construction here is JSON validation; no author code executes. Four fresh
Astra/xhigh calls with visible transcript and development feedback,20min/call,
four-hour study guard,no automatic retry,existingCodexusage,no hard dollar/output
token cap,no API-key fallback. Expected30-60minutes, uncertain. GPU used only for
frozen-model feature caching/evaluation; key fitting CPU. No large checkpoints:
one small final key+fit trace per fit,shared parent referenced by hash.

Source gate and bound runtime/model/data. Atomic essential records with retried,
nonfatal status writes. Base model hash before/after; frozen keys checked after
evaluation. Abort on finite/bounds/infrastructure failure. No extra run is
automatically authorized, including cross-concept generalization or fullARC.
