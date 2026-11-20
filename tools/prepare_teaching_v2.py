"""Prepare a fresh replacement pin, preserving historical evidence; no model calls."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from run_executive_teaching import sha
from skillforge.executive_teaching import teacher_prompt
from skillforge.student_control import prompt
from skillforge.codex_one_shot import preflight
from skillforge.reliable_json import save


def main():
    old=ROOT.parent/'WSPL-executive-teaching-v1/results/executive_teaching_v1'
    manifest=old/'approved_manifest.json'
    if sha(manifest)!='52e43287bc87b5ff51315bbd395a9cca1f6a84a082342b996de4c45666d5a539':
        raise ValueError('Historical manifest identity changed')
    spec=json.loads(manifest.read_text());spec.pop('files');spec.pop('schema',None)
    c=spec['runner_config'];resources=c['resources']
    for path,h in resources.items():
        if sha(path)!=h:raise ValueError('Historical resource changed '+path)
    archive={str(p.resolve()):sha(p) for p in old.rglob('*') if p.is_file()}
    resources.update(archive)
    for name in ('EXECUTIVE_TEACHING_V2.md','EXECUTIVE_TEACHING_ARRIVAL_REPAIR.md','EXECUTIVE_TEACHING_V1.md'):
        p=ROOT/'docs/wspl'/name;resources[str(p)]=sha(p)
    spec['model_id']='executive-teaching-v2'
    c['sim_worker']=str(ROOT/'tools/teaching_sim_server.py')
    c['historical_archive']=dict(root=str(old),files=len(archive),completed_calls=40,interrupted_calls=1)
    public=json.loads((ROOT/'data/teaching_preflight/public_fixture.json').read_text())
    preview=ROOT/'data/teaching_v2_prompt';preview.mkdir();(preview/'empty').mkdir()
    preflight(c['cli'],c['catalog'],c['instructions'],preview,teacher_prompt(public))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('C:/Users/kevin/skillforge-runtime/models/Qwen2.5-Coder-1.5B-Instruct',local_files_only=True)
    rendered=tokenizer.apply_chat_template([dict(role='user',content=prompt(public['observation'],public['documentation']))],tokenize=False,add_generation_prompt=True)
    n=len(tokenizer(rendered,add_special_tokens=False)['input_ids'])
    if n+96>4096:raise ValueError('Prompt context exceeded')
    save(ROOT/'experiments/source_pins/executive_teaching_v2.spec.json',spec)
    report=dict(status='prepared',approved=False,resources=len(resources),historical_files=len(archive),prompt_tokens=n,new_model_calls=0)
    save(ROOT/'docs/wspl/teaching_v2_preparation.json',report)
    print(json.dumps(report))


if __name__=='__main__':main()
