# Completed fresh Astra native Breakout authoring screen

November 3, 2026. Three separate fresh Astra conversations each produced a usable
native neural policy in one final reply. All three policies completed the
prespecified 1,000-episode evaluation, and the study and source receipts report
complete. This is an exploratory supplied-rule authoring result on one task.

## Frozen design and result

Requested proposer: gpt-6-astra, xhigh reasoning, through ChatGPT-authenticated
Codex CLI 0.164.0-alpha.11.1. Each received the same original CSAIL Breakout task
prompt from upstream commit 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f, including
rules, environment source, network specification and the native compact edit
language. No reference answer or weights, score feedback, gradient training,
repair round or best-of selection was provided. One final reply per draw, with
all three retained. The shared zero-initialized architecture has 20,486 FP32
parameters: 600 -> 32 -> 32 -> 6, two tanh hidden layers and linear logits.

| Draw | Mean return | Episode SE | Nonzero parameters | Valid / skipped edits | Horizon truncations | Collection seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 484.756 | 4.469 | 3,339 | 193 / 0 | 804 / 1,000 | 580.391 |
| 2 | 541.818 | 5.000 | 1,032 | 118 / 0 | 775 / 1,000 | 681.953 |
| 3 | 510.042 | 7.755 | 1,153 | 91 / 0 | 521 / 1,000 | 517.500 |

The descriptive mean of the three policy scores is 512.205; sample standard
deviation across policies is 28.592. Episode SE measures evaluation uncertainty
conditional on each fixed policy, not uncertainty across authoring draws. There
are only three authoring draws, not 3,000 independently authored policies.

Every policy used the same 1,000 environment/action seed pairs (510000+i /
610000+i), native six-action sampled softmax, sticky actions 0.1, difficulty
ramping and a 10,000-step horizon. All returns are retained, including truncated
episodes. The cap limits interpretation of unrestricted episode returns.
Local evaluation used one CPU math thread; no local GPU or optimizer updates.
Hosted proposer's compute is not measured by the local CPU accounting.

For context, the separate released-artifact replay scored 511.828 +/- 7.605 SE,
and its uniform zero-weight control scored 0.551 +/- 0.0248. The upstream report
was 517.411 +/- 7.417. Those use different episode seeds, so none is a matched
policy contrast or evidence of superiority/equivalence. See the
[released-artifact replay](BREAKOUT_REPLAY_RESULTS.md).

## Usage, isolation and complete attempt history

Run elapsed time: 2,468.609 seconds (41.1 minutes). Reply collection totaled
1,779.844 seconds (29.7 minutes); the remainder includes evaluation and overhead.
CLI-reported usage across the three completed replies:

| Metric | Tokens |
| --- | ---: |
| Input | 10,020 |
| Cached input, included in input | 3,200 |
| Output | 65,775 |
| Reasoning output, included in output | 55,638 |

Reasoning is not added to the output total a second time. Dollar cost, remaining
account capacity and backend compute were not measured. Third-draw prompt-prefix
caching is recorded; conversations have distinct thread IDs and earlier replies
were not supplied to later draws. Generation sampling seeds were not controlled.

The restricted CLI preview offered only request_user_input; all recorded turns
used zero tools. Each contains one agent message and one completed turn, with
the saved answer matching the final event. User rules, AGENTS content, skills,
memory and repository files were excluded by the configured harness. The preview
is local client evidence, not a server-side context/model attestation. Requests
explicitly select Astra. An auxiliary unknown gpt-5.6-luna metadata warning
remained in CLI stderr; it is preserved, not treated as proof of model identity.

Before this successful run, one invocation under v1 was aborted after a startup
warning was emitted as an error item: no completed reply or evaluated policy,
usage unknown. Thus the full history is four harness invocations, three completed
replies. No hidden retries or replacement within the successful three-draw run;
bounded CLI transport retries were permitted by the documented harness. See the
[failed-attempt archive](astra_native_v1_startup_failure.json) and
[approved repair](ASTRA_BREAKOUT_NATIVE_V2.md). Preparation time and the aborted
attempt's 20.094 seconds are excluded from the successful run elapsed time.

## Audit and preservation

The completion audit passed, without model calls, fitting or additional game
steps. It verifies all 65 pinned source files, bound resources/executable/core
dependencies, the complete source receipt and three constructor calls, every
episode index/seed/action count/termination flag, and independent standard-library
recalculation of all score summaries. Reinterpreting each final reply reconstructs
every saved FP32 NPZ tensor bitwise. This checks parser-to-artifact reproducibility;
it is not an independent semantic proof of the author's intended algorithm.

All 3,000 raw episode records, three raw replies/event streams, prompt previews,
commands, usage receipts and one final NPZ per draw remain in the isolated local
Skillforge-astra-breakout-v2/results/astra_breakout_native_v2 archive. There are
no added checkpoint copies. The compact [audit/results](astra_native_v2_results.json)
binds raw-file hashes; [all final replies](../../artifacts/astra_breakout_native_v2/README.md)
are also archived byte-for-byte in Git. Credentials and binary weights are not.

Approved source manifest:
`73bf70a24e92b978831aeca4998c44d049b6b2f39b4dda9544c5409b02b7fc46`.
Execution checkout: feae13d. Historical launch evidence remains in
[the startup record](astra_native_v2_launch.json); it is superseded for current
status by these completed results. The frozen protocol files are preserved.

To repeat the read-only audit with the original dependencies and local archive:

```sh
python tools/audit_astra_breakout_native_v2.py /path/to/Skillforge-astra-breakout-v2
```

## Interpretation

This establishes fresh, one-response native neural weight authoring with useful
empirical performance in this supplied Breakout task. The model can express a
controller directly as bounded edits to a small fixed network. Three draws offer
initial evidence of repeatability under this prompt and harness.

It does not establish an advantage for WSPL, superiority over gradient learning,
formal correctness, transfer, novel-task generalization, robotic control or
recursive twin improvement. The model received extensive game knowledge; weights
were authored from those supplied rules rather than learned from experience in
this screen. Prior training exposure to the public game/research release cannot
be excluded. A matched WSPL/code/native comparison and independent gradient
baseline are separate future experiments requiring their own approval.
