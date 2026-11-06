# Engineer a reusable neural spatial capability

Objective: install reliable one-cell marker-directed movement into a frozen
copy-trained SmolLM2-135M. Generalize to unseen connected shape families and larger
objects; retain copying. Do not optimize merely a few corrected examples.
Choose neural module structure, placement, initialization and the available
training recipe. Your final artifact is a single fixed set of neural modules.
No Python/grid solver/lookup table runs at inference. No reserved outcomes shown.

Model:30layers,width576,9queryheads,3KVheads,64dimensions/head,FP32. Base weights
are frozen. Projection q/o maps576->576;k/v maps576->192. Modules attach as
`projection(x) + branch(x)` at named sites `0_q`..`29_o`. Each branch sees only
the native projection input, at every token, without a task/position dispatcher.
Default120branches are rank16 affine low-rank paths initialized with zero writes
(1,891,200 trainable parameters including biases). At most2.5M total parameters,
120branches,250k parameters/branch,16nodes/depth8/live8192 per branch.

Permitted branch DAG operators:affine,ReLU,tanh,identity,add,scale,concat,slice.
Arbitrary valid DAGs within that envelope are permitted equally to both interfaces.
The host applies them in PyTorch and trains all branch tensors. This is a bounded
module/training service, not unrestricted model-code execution. Full transformer
attention is outside WSPL; host attachment is shared. Module correctness does not
certify task semantics. No direct base-weight or vocabulary-head edits.

Domain:6x6 grids,background0,object color1..8,marker9. Marker lies outside the
object bounding box and aligned with its row/column span, defining one direction.
Move exactly one cell toward it without collisions/out-of-bounds. Copying returns
the input unchanged. Input/output is row-major digits. Prompts supply this rule;
there are no worked demonstrations or scratchpads. Direct greedy output consumes
exactly36tokens (16for4x4copy), unrestricted vocabulary, no retries/repair.

Common train pool:8,192movement grids,4,096random6x6copies,4,096random4x4copies.
Development:512movement,256copy6,256copy4. Whole canonical shape families are
excluded from training/development for final transfer. Final evaluation also
tests larger objects and two applications of the SAME frozen model. You cannot
inspect final fixtures or scores. Packet contains fixed representative train/dev
examples; the host training service has the full common train pool. You cannot
run custom Python over the full data or full model in this screen.

Per arm:four responses, at most8,192updates and3,600 synchronized training seconds
cumulative. No reset restores a budget. Each accepted response runs128..2,048
updates of batch4, subject to remaining budget, then receives development results.
First512cumulative updates use only copy4/copy6 equally, for common I/O learning.
Afterward copy_fraction controls total copying, equally split by size. Movement
and copying are sampled with replacement. Plain AdamW,zero decay,clip1,32update
LRwarmup at each chunk; optimizer moments reset each chunk in EVERY arm, including
the ordinary-gradient control. There is no checkpoint/best selection; last state
is retained. Rejected proposals and FINAL consume a response. No automatic retry.

Available recipe fields:lr in[1e-5,.005];copy_fraction in[.125,.75];
changed_cell_weight in[1,8];steps integer[128,2048]. Full-vocabulary answer-only
cross entropy; changed_cell_weight upweights movement cells differing from input.
This uses training labels only; no movement calculation occurs during inference.
Default recipe:.001,.25,1,2048. You may propose numerical optimization in NumPy
for module construction, but there is no live model/gradient access in the sandbox.
The shared service supplies actual full-model adapter training after construction.

Respond with one fenced Python program or FINAL. Provided construction API:
`names()` lists current sites; `g,a=load(name)` returns a copied graph and FP32
NumPy arrays; `save(name,g,a)` replaces a module; `add_lora(name,rank=16,seed=701)`
reinitializes/adds it; `remove(name)` drops it; `recipe(lr=...,steps=...)` updates
the recipe. `np` and `packet` are provided. Print compact diagnostics. Imported
Torch/GPU/model/data tools are unavailable. Use only the documented API and NumPy;
do not attempt to bypass interface boundaries or inspect library files.

Graph schema:
`{'kind':'dag','inputs':{'x':576},'nodes':[...],'outputs':{'y':'node:up'}}`.
Node has exactly id/op/inputs/params/attrs. Affine example:
`{'id':'down','op':'affine','inputs':{'x':'input:x'},'params':{'weight':'A','bias':'a'},'attrs':{'layout':'out_in'}}`.
ReLU/tanh/identity use inputs{x:signal},empty params/attrs. Add/concat use inputs
{a:signal,b:signal},emptyparams/attrs. Scale uses inputs{x:signal},params{gain:G},
emptyattrs and G shape[1]. Slice uses inputs{x:signal},emptyparams,attrs{start,stop,step}.
All signals are input:x or node:ID; sharing a tensor name explicitly shares storage.
Use save after editing arrays/graph. Equal native and WSPL branches deploy identically.

Additional API ONLY in native-plus-WSPL arm:
`wspl(name,source,patches=None,bindings=None)` executes a checked WSPL transaction
against the current branch. `$PARENT` substitutes its exact artifact hash.
Example:
`wspl('0_q','wspl 0.2; target base=artifact("base",sha256="$PARENT"); edit base as candidate mode=fixed { b[:]=0; }')`.
Native tools remain available. Graph mode uses budget="module"; patches map names
to explicit tensor-artifact documents, bindings supply construction tensors.
The language's graph surgery reference is appended for this arm. Changes are
staged, identity-checked and replayed; receipts record actual WSPL use. Native arm
must not invoke/import WSPL editing or checking APIs. Either arm may build the
same graph/tensor result via ordinary native operations.

Sandbox:8GiB memory,240CPU seconds/270wall seconds,256MiB artifact limit,no network,
read-only package runtime and request. Only current modules and packet exposed.
Numerical fitting/training is allowed and will be reported; no gradient-free claim.
No tool calls/questions. All prompts/programs/feedback/usage are archived.
