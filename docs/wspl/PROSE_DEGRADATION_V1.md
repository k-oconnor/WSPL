# Simple prose degradation check

Compare the frozen copy-trained SmolLM foundation with its saved WSPL 0 movement
adapter enabled. No training, author calls, adapter edits, routing or recipe search.

Use the first 64 documents in the existing FineWeb-Edu validation partition.
Decode the existing corpus tokenizer, remove only its synthetic BOS/EOS boundaries,
and verify each full document against its recorded UTF-8 SHA-256. Re-encode with
SmolLM's tokenizer and score the first at most 513 tokens per document: at most
512 next-token targets, no cross-document transitions. Report pooled token NLL,
perplexity and paired change. The sample is held out from local grid adaptation;
no claim of absence from SmolLM's pretraining data.

Also retain all three fixed greedy continuations, at most 64 new tokens each,
for both conditions. Prompts: "The water cycle describes how", "To make a cup of
tea, first", and "The purpose of a public library is". No selection or retries.
Use the same FP32 eager model and exact saved adapter attachments as the ARC
evaluation. Check frozen base and adapter hashes. This is a small descriptive
degradation check, not a new capability benchmark or WSPL comparison.

Expected a few minutes, 20-minute guard, GPU inference only. No model copies.
Source approval required by AGENTS.md; nothing automatically follows this run.
