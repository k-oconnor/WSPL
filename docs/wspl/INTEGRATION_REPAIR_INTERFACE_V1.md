# Repair integration of a learned spatial adapter

The frozen copy-trained SmolLM2-135M foundation has learned movement adapters
which solve grids well but damage ordinary language. You start with the exact
trained WSPL 0 artifact, not random parameters. Preserve its reusable spatial
skill while recovering ordinary prose next-token prediction. Seek a domain-level
repair, not patches to the supplied examples. The same fixed adapter is enabled
for both domains. No external task detector, prompt changes, solver, or switch.

You have four responses total, each one fenced Python program or FINAL. Each
valid program edits the current artifact and receives acquisition-only movement,
copying and prose-loss feedback. Last accepted state is final; rejected responses
consume budget. There is NO host training. Preserve the recipe fields unchanged.
You may inspect/edit arrays and perform NumPy calculations in the sandbox. No
full-model, gradient, network, reserved-data or tool access. No automatic retries.

Baseline feedback includes strengths 1,.75,.5,.25 applied to every whole branch.
The scalar control is selected using acquisition results only: lowest prose NLL
retaining at least90% of original movement successes and losing at most one copy.
Simply globally weakening the adapter is permitted, but does not establish an
advantage over attenuation. You can instead change branch weights, strengths,
placement and supported graph structure, within the same 2.5M parameter budget.

Model:30layers,width576,9Q/3KVheads,head64. Each named site 0_q..29_o attaches
projection(x)+branch(x). q/o output576;k/v output192; all inputs576. Branches see
native representations at every token. Base model and vocabulary head frozen.
At most120 branches,250k parameters/branch,2.5M total,16 nodes,depth8,live8192.
Allowed ops affine,relu,tanh,identity,add,scale,concat,slice. No task metadata.

API: names(); g,a=load(name) returns copied graph and NumPy FP32 arrays;
save(name,g,a); remove(name); add_lora(name,rank=16,seed=701) replaces/adds a branch.
np and packet are supplied. Native editing and WSPL both available. Do not change
recipe. Use only these documented APIs; do not inspect host/library files.

Graph: {'kind':'dag','inputs':{'x':576},'nodes':[...],'outputs':{'y':'node:up'}}.
Each node has exactly id,op,inputs,params,attrs. Affine example:
{'id':'up','op':'affine','inputs':{'x':'node:down'},'params':{'weight':'B','bias':'b'},'attrs':{'layout':'out_in'}}.
Scale uses inputs{x:signal},params{gain:tensor_name},attrs{} and gain shape[1].
relu/tanh/identity use inputs{x:signal},empty params/attrs. add/concat use
inputs{a:signal,b:signal};slice uses attrs{start,stop,step}. Signals input:x or
node:ID. Tensor sharing within a branch is explicit through parameter names.

wspl(name,source,patches=None,bindings=None) runs an identity-checked transaction.
$PARENT substitutes the exact current artifact hash. Example:
wspl('0_q','wspl 0.2; target base=artifact("base",sha256="$PARENT"); edit base as candidate mode=fixed { b[:]=0; }').
Graph edits use budget="module". The graph reference follows this guide.

Acquisition:32movement grids,16copy6 grids,32prose documents. Final evaluation
uses separate grids and prose, never exposed during authoring. Movement uses
the same supplied one-cell marker rule and unrestricted greedy 36-digit output.
Prose loss uses first at most512 next-token targets per document. No ARC feedback.

Descriptive target: recover at least half the prose NLL gap to the foundation,
retain at least90% of original successes separately on reserved novel and larger
shapes, and lose at most one copy. Separately compare against scalar attenuation.
All programs, construction receipts, feedback and model usage will be archived.
Sandbox8GiB,240CPU/270wall seconds,256MiB artifact,no network. Return no tool calls.
