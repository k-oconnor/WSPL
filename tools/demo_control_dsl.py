"""Replay the supplied two-object engineering fixture; no learned-model evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from skillforge.control_dsl import Executor, Workcell, compile_program
from skillforge.controller_author import KINDS, make_spec, teacher, construct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jam-second', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing trace')
    source = 'transfer a to tray\ntransfer b to buffer\nmove_to home\n'
    world = Workcell((-.7, -.6), {'a': (-.25, .1), 'b': (.3, -.3)},
                     jammed=frozenset({'b'}) if args.jam_second else frozenset())
    points = {'tray': (.6, .55), 'buffer': (-.5, .55), 'home': (-.7, -.6)}
    specs = {k: make_spec(k, 20) for k in KINDS}
    blocks = {k: construct(teacher(s), s) for k,s in specs.items()}
    executor = Executor(compile_program(source, world.objects, points), points, blocks)
    result = executor.run(world)
    result.update(program=source, engineering_fixture=True,
                  blocks={k: b.to_dict() for k,b in blocks.items()}, final=world.observe())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','reason','ticks','completed_instructions')}))


if __name__ == '__main__':
    main()
