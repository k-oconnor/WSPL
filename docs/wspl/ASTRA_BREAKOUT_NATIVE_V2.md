# Fresh Astra native screen: startup repair

Prepared replacement, not approved or launched. The scientific design and limits
in [v1](ASTRA_BREAKOUT_NATIVE_V1.md) remain: three fresh Astra/xhigh proposals,
original prompt/edit syntax and tanh architecture, 1,000 paired episodes per
evaluable policy, no feedback, 1,200-second invocation/7,200-second study guards,
existing ChatGPT Codex usage without a dollar/output-token cap.

## Preserved failed attempt

Kevin approved v1 manifest
`07eeeeda72b5bfc91ecb07ba94c4ee0593fcc3caee50577dde2542f2c87f9f33`.
It was registered and launched in the isolated v1 checkout on November 3 at
21:05:42 UTC. The first invocation emitted an item of type error containing the
CLI's under-development-feature warning. The strict collector would reject this
event. The controller stopped the invocation; the study ended incomplete after
20.094 seconds, one attempted draw, two not attempted, zero completed replies,
zero evaluated policies, no reported final token usage. Backend usage for that
aborted invocation is unknown. Both workers exited. The source receipt is
complete for the controlled exit; the study result is incomplete.

Preserve that attempt's raw logs, manifest and denominators; do not overwrite or
exclude it from the authoring history. This replacement proposes three further
invocations, additional to the one aborted invocation. It does not reclassify the
failed draw as successful or unused.

## Narrow repair

- Pass the documented `suppress_unstable_features_warning=true` configuration
  override, retaining host skill discovery isolation and strict error rejection.
- Pass `--model gpt-6-astra` explicitly as well as the existing model setting.
  The first CLI stderr also warned about unknown gpt-5.6-luna metadata. That
  warning alone does not establish which model handled the interrupted request;
  the replacement removes ambiguity in the CLI model argument.
- Check event lines while the subprocess is running. Any tool/error/unknown
  event terminates collection promptly, rather than waiting for the final audit.
  Final fresh-thread/one-turn/answer/usage checks still apply.

Fake-process regression reproduces the startup error and verifies prompt abort.
No further model requests are used for testing this repair. The original-prompt
offline preflight must pass again. Architecture, data, evaluation seeds,
parser, authentication, proposer effort, budgets and failure policy are fixed.
New source requires a new exact digest approval under AGENTS.md.

Official warning setting:
https://learn.chatgpt.com/docs/config-file/config-reference
