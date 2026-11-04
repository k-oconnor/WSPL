"""Export frozen public prompts only; no model requests or game evaluation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def prompts(native_text,guide):
    game=native_text[native_text.index('THE GAME:'):native_text.index('THE NETWORK (')]
    common=('YOUR TASK: author a fixed neural Breakout policy through the selected interface. '
            'This is a one-response authoring task with no tools or evaluation feedback. '
            'Briefly explain your design, then provide only the selected final candidate block.\n\n'
            +game+'\nSHARED COMPARISON INTERFACE PACKET\n'+guide)
    return {arm:common+'\n\nSELECTED ARM: '+arm+'\n' for arm in ('weights','code','wspl')}


def main():
    source=ROOT/'data/csail_breakout_reference_v1/exp5_write_from_scratch/runs/breakout/message.txt'
    guide=ROOT/'docs/wspl/BREAKOUT_MATCHED_INTERFACE_V1.md'
    target=ROOT/'data/breakout_matched_v1';target.mkdir(exist_ok=False)
    exported=prompts(source.read_text(encoding='utf-8'),guide.read_text(encoding='utf-8'))
    records={}
    for arm,text in exported.items():
        path=target/(arm+'.prompt.txt');path.write_text(text,encoding='utf-8',newline='\n')
        records[arm]=dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    bundle=dict(format='breakout.matched_prompts.v1',prompts=records,
                original_prompt_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                shared_guide_sha256=hashlib.sha256(guide.read_bytes()).hexdigest(),
                proposer_responses_used=False,task_solution_included=False)
    (target/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
