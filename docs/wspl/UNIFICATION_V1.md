# WSPL 0.3: definition, editing and composition

Implemented experimental frontend; legacy artifacts and frontends remain readable.
Entry point: `skillforge.wspl_language.compile_source`. No learned result is
claimed for this version until the separately approved repeat completes.

## One source unit, three execution stages

WSPL 0.3 keeps typed Core PWA `module` definitions as the canonical scalar
function language. It connects their compiled artifacts to the tensor/graph
editor and adds typed task calls to a host-bound module library. Construction
and runtime composition are different stages, with explicit identities between
them. This is not a new neural architecture or learned runtime router.

```wspl
wspl 0.3;
module Drive {
  input error: Real[-2,2] normalized_error;
  output command: Real[-1,1] normalized_command;
  command = clip(error / 2, -1, 1);
}
target base = compiled(Drive);
edit base as tuned mode=fixed {
  b1[0] += 1/100;
}
```

The module uses the existing typed parser, scalar IR and compiler. `compiled`
bridges its rational network to the editor's FP32 artifact, retaining port/type
metadata in the source module and recording coefficient rounding separately.
The returned compilation contains both the original module and the candidate;
an edit does not silently replace the module or inherit its contracts. Installing
an edited module into a runtime library remains an explicit host admission step.

High-level functions such as `clip`, `min` and `max` belong in module definitions.
The compiler lowers them into affine/ReLU operators. Existing `circuit`/`place`
forms remain available for explicit neuron placement, with their stricter layout
requirements. They are a low-level placement interface, not a competing typed
module language. No new implicit relays or capacity changes are introduced.

## Typed task composition

```text
wspl 0.3;
use robot = library("planar", sha256="ACTUAL_LIBRARY_SHA256");
task Mission {
  robot.transfer(parcel, tray);
  robot.move_to(home);
}
```

The digest is the full actual hash of the host library descriptor. It binds all
six module artifact hashes, their port types, supplied numerical specifications,
the skill signatures and the runtime profile. A mismatched binding is an error;
`library` does not load a path, download code or run a proposal script.

| Skill | Signature | Runtime behavior |
|---|---|---|
| `move_to` | `(ObjectOrPoint)` | Movement, proximity and safety modules with observed completion checks |
| `grasp` | `(Object)` | Close/holding modules with geometric contact checks |
| `release` | `(Object)` | Open/holding modules with observed release checks |
| `transfer` | `(Object, Point)` | Four ordered primitive calls; no new neural weights |

The compiler checks names, arity, object/point types, library identity and expanded
instruction limits. It produces typed `Call` IR directly, without translating
back to the old action text language. The existing stateful interpreter executes
that IR. Task safety/failure handling remains host runtime behavior, not a theorem
about an arbitrary neural controller. The runtime implementation is source-pinned
for experiments in addition to the descriptor identity.

## Accepted top-level grammar

```ebnf
program = "wspl" "0.3" ";" { module | use } { editor_declaration } [ task ] ;
use = "use" name "=" "library" "(" string "," "sha256" "=" string ")" ";" ;
task = "task" name "{" call { call } "}" ;
call = library_alias "." skill "(" name { "," name } ")" ";" ;
```

Exactly one task is allowed, last in the source. It binds one runtime library
identity. Empty source units and empty tasks fail. Module declarations retain the
Core PWA rules in [IMPLEMENTED_SYNTAX.md](IMPLEMENTED_SYNTAX.md). Editor declarations
retain the existing target/constant/circuit/region/edit/describe rules, with the
additional `target name = compiled(ModuleName);` binding. A source has at most one
editor target/transaction. Module/import declarations precede editor declarations.
Standalone task files need no embedded module weights because their imported
library binds the compiled modules explicitly.

The task grammar has no runtime loops, branches, dynamic loading or replanning.
The current skill ABI is the planar observed-state profile. General skill
definitions and new physical robot adapters are future work; this version does
not claim arbitrary neural module composition or a new LTLf compiler. The earlier
controller experiment's supplied LTLf layer and the executive's ordering task
remain distinct evaluation settings.

## Compatibility and evidence

Legacy Core modules and WSPL 0.2 source continue through their existing APIs.
Adding `wspl 0.3;` before a Core module preserves the rational compiled artifact
identity. Old executive text remains an archival format. New executive programs
use typed calls. Engineering checks compare full runtime traces and final scores
across both paths, including bounded jammed-grasp failure. No historical reply or
checkpoint is rewritten, and compatibility does not imply that an old author has
learned the new task syntax.
