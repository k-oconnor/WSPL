# CSAIL replication and WSPL product win conditions

November 3, 2026. Kevin endorsed selected CSAIL replications and WSPL extensions,
with startup success defined by customer value rather than a requirement to beat
reported game scores. Start with released Breakout artifact replay, fresh Astra
authoring, a matched WSPL extension and then learned-policy repair. The Boolean
screen serves as harness validation for this sequence.

## Reference experiments

The authors' [MinAtar protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp5_write_from_scratch/README.md)
uses a 600-32-32-6 tanh policy, one Astra reply, no development feedback and a
compact weight-edit language. Evaluation uses MinAtar 1.0.15, sampled actions,
sticky actions 0.1 and 1,000 episodes with a 10,000-step limit. PPO checkpoints
are not included; published PPO scores remain external references until reproduced.
Their [Kinetix repair protocol](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp4_fix_car/README.md)
also includes tanh layers. These source protocols were checked November 3.

Keep artifact replay, native Astra replication and matched WSPL extensions
separate. Replication retains the original architecture, edit interface,
information and tool/feedback restrictions. Extensions compare WSPL, ordinary
code and competent native weight editing at a common supported architecture.
A ReLU adaptation is labelled as an extension, not an exact tanh replication.
Match model settings, tool access and feedback within each comparison.

Our current compiler/verifier supports ReLU graphs, not the original tanh models.
Game returns are empirical; supported graph proofs do not prove winning or
simulator-wide safety. Public task/solution exposure and fresh authoring draws
must be logged. Do not hide architecture expansion in a language comparison.

## Win conditions

These numerical targets are prospective internal choices. Finalize them against
task/customer requirements and freeze them before final trials; no result or
customer commitment is established yet.

| Gate | Proposed condition | Required evidence |
| --- | --- | --- |
| Useful behavior | Meet an absolute usefulness floor and retain at least 90% of each matched baseline's mean Breakout return; repair gets a separately frozen success threshold | Fresh evaluation conditions, every proposal counted, variability across independently authored policies |
| Required assurance | All required supported graph and preservation contracts PROVED on every admitted artifact; UNKNOWN never authorizes commit | Non-vacuous domains, identities, replayed counterexamples, transaction record and explicit numerical scope |
| Workflow benefit | At most half the active engineering plus review time of the matched ordinary-code/current workflow, while passing the first two gates | Specification, integration, failed proposals, diagnosis, revision, verification and review costs included |
| Buyer commitment | One external team supplies a permitted real change request and commits budget to a paid pilot | Identified user/budget owner, agreed deliverable and acceptance criteria, actual commercial commitment |

Choose the absolute usefulness floor in feasibility work before final trials;
exclude a degenerate baseline that barely exceeds zero/random behavior. The 90%
return ratio is a proposed game-pilot tolerance, not an acceptable regression in
every customer domain. Freeze absolute safety/retention requirements separately.
A customer may require full performance retention or a tighter margin.
Requirements are frozen before trials; an unsupported required check leaves
qualification unresolved rather than being dropped from the acceptance criteria.

The pilot reports point estimates and uncertainty, not general noninferiority.
Confirmation must freeze independent authoring replicates, task coverage, nested
episode analysis and comparison-wise error control before calls. More episodes
from one policy do not create independent authoring trials. A noisy apparent
90% ratio does not establish parity.

## Product benefits and controls

Similar performance with faster reviewed changes is the primary proposed benefit.
Other candidates are fewer paid proposals, lower total cost per qualified change,
preserved behavior after repair, or evidence a reviewer would otherwise need to
create manually. Select one primary benefit before final evaluation and report
the rest as secondary; do not select the winning metric after seeing results.

An evidence-focused product can be valuable when customers require contracts.
Declare that workflow separately and test whether the reviewer completes
acceptance with less work than ordinary code plus the same verifier. More reports
or PROVED labels alone do not establish customer value. A benchmark success without
a buyer is a technical milestone, not commercial validation.

Keep the native compact edit interface as a competent baseline. Give ordinary
code and native editing the same applicable verifier access in feedback-on arms.
Count compiler/solver work, billed reasoning and prompt tokens, environment calls,
setup and human review. Disclose added depth, parameters and latency; augmentation
and in-place repair remain separate tracks. Recheck changed artifacts.

Retain the parent on rejected or unresolved transactions. Test malformed edits,
stale identities, interface violations, false preservation and exhausted budgets.
Keep all setup failures and negative outcomes. Zero failures on a test panel is
not a universal soundness result.

## Decision and execution boundary

Technical progression requires useful behavior, required assurance and a credible
predeclared workflow benefit. Commercial validation also requires the buyer gate
and evidence on a real workflow. Reconsider the language interface if ordinary
code with the same tooling is equally effective and easier. A review/transaction
product can be tested separately if that is where users find value.

RTI remains a later hypothesis; domain improvement alone does not show better
editing ability. These product gates do not establish recursive improvement.

The first fixed-artifact replay is now
[complete](BREAKOUT_REPLAY_RESULTS.md): released-policy mean 511.828 versus
upstream 517.411 and uniform control 0.551, with 1,000 episodes per arm. The
[approved protocol](BREAKOUT_REPLAY_V1.md) used pinned resources and isolated CPU
dependencies. This is artifact replay, not fresh Astra authoring.
The [fresh native authoring screen](ASTRA_BREAKOUT_NATIVE_RESULTS.md) has since
completed three policies (484.756,541.818,510.042 mean return). The
[matched ReLU extension](BREAKOUT_MATCHED_V1.md) is prepared with native edits,
restricted Python PWA source and WSPL, three fresh draws each. The exact source
and nine invocations are now user-approved and the run has launched; see its
[startup record](breakout_matched_v1_launch.json). No comparison result exists yet. Trained PPO references
remain unprepared. The $150 estimate covered the Boolean screen, not PPO
rebuilding or this expanded benchmark. No outreach is sent here.
