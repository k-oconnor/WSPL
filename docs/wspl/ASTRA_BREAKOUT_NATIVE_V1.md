# Fresh Astra native Breakout authoring screen

Status: prepared, unapproved; no fresh model response collected.

This exploratory screen asks whether fresh one-shot Astra proposals can construct
useful native Breakout weights. The completed released-artifact replay validates
the evaluator separately. This screen does not compare WSPL, train a baseline,
prove tanh controllers, or establish a commercial win.

## Frozen design

- Three independent fresh Codex conversations, sequentially, using the existing
  ChatGPT login, `gpt-6-astra`, reasoning `xhigh`, standard speed (fast modes off).
  No API-key fallback, extra credit purchase or harness retry is authorized.
  CLI-managed bounded transport retries may occur; there are at most three
  harness invocations, not a guarantee of three HTTP requests.
- Give each the original CSAIL Breakout `message.txt`, unchanged, from upstream
  commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Neutral instruction:
  `Answer the user's request.` No reference answer or released weights supplied.
- Native 600 -> 32 -> 32 -> 6 tanh network, 20,486 FP32 parameters, initially zero.
  Read the last complete edits fence. Apply ordered data-only assignments/slices
  and +=/-=/*=; record and skip invalid statements. Never execute proposal code.
  Empty/all-skipped complete fences remain evaluable zero policies. Missing
  fences, excessive output or invalid collection remain explicit failures.
- Three complete proposals, including zero policies, are evaluated separately on
  the same 1,000 environment/action seed pairs: 510000+i / 610000+i, i=0..999.
  Native six-action sampled softmax, sticky actions 0.1, difficulty ramping,
  10,000-step horizon; all horizon truncations retained. CPU, one math thread.
  Generation receives no scores, environment interaction or repair feedback.
- Report every draw's parse outcomes, tensor/NPZ identity, mean/SE, episode
  outcomes, CLI token usage and time. Keep the planned denominator of three,
  including invalid, failed and unattempted proposals. No best-of selection,
  checkpoint tuning or automatic continuation. Shared episodes are paired
  evaluation conditions, not three thousand independent authoring trials.
- One immutable final policy per draw, no checkpoint proliferation. Preserve
  prompts, raw event streams, stderr, exact replies and incomplete results.

## Isolation and limits

CLI 0.164.0-alpha.11.1 is executable-hash bound. Each invocation ignores user
configuration/rules, uses ephemeral sessions, disables the daemon, agents,
skills, memories, MCP, apps, web, shell, browser, image and editing tools, and
removes parent thread/API-key/provider environment variables. An empty workspace
and zero AGENTS byte budget prevent repository context from entering the prompt.
Authentication remains in the user's existing home; credentials are never copied
into the repository or logs. Live model entitlement and account capacity have
not been tested by a model request.

A hash-bound local copy of cached Astra client metadata changes only tool
presentation: experimental tools become empty, apply_patch_tool_type becomes
null, and tool_mode becomes default. This is a client harness choice, not a model
weight change. The offline rendered prompt exposes only request_user_input,
which cannot be used in this experiment. Permissions/default collaboration and
environment context remain CLI boilerplate. The renderer is checked before each
call with an empty unauthenticated configuration home; this preview is not a
server-side attestation. Actual calls use ignore-user-config/ignore-rules.
Any tool item, additional turn, unknown event, failed turn, missing usage or
answer mismatch invalidates collection and stops the queue without retry.
The client still advertises the question tool, so this is a restricted CLI
harness, not literally a tool-free API request. No filesystem/action tools are
offered in the checked preview.

Per-invocation wall guard: 1,200 seconds. Whole screen: 7,200 seconds. A stop
control terminates collection or evaluation and retains partial logs. Time guards
bound local execution; cancellation cannot guarantee backend accounting stops
instantaneously. There is no enforceable dollar or output-token cap in this
CLI path. CLI-reported tokens are recorded without inventing a dollar cost or
remaining balance. The previously proposed $150 API budget is not approval here.
No new paid API credits are requested. Existing ChatGPT/Codex usage is consumed.

## Engineering checks and approval

The data-only parser reconstructs the released six tensors bitwise: 92 valid
statements, none skipped, 815 nonzero parameters. That read-only audit does not
give the new proposer the released answer. Fake CLI tests cover capture, usage,
tool/turn rejection, timeout and isolation. Fake environment integration covers
three fresh policies, paired seeds, invalid proposals and fail-fast denominator
preservation. The original-prompt offline rendering passes; no model generated.

Launch only via source_consistency.py after the exact candidate manifest digest
is approved and independently registered. Source, protocol, prompt, metadata,
instructions, executable and scientific dependencies are bound. Use the isolated
Skillforge-astra-breakout-v1 checkout; no source changes during execution.

Official CLI/config references:
https://learn.chatgpt.com/docs/cli/reference
https://learn.chatgpt.com/docs/config-file/config-reference
