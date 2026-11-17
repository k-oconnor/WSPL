"""Read-only artifact/token audit and unapproved manifest specification."""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from skillforge.executive_study import load_library
from skillforge.wspl_language import compile_source,library_document,digest as object_digest
from skillforge.unified_executive import prompt,teacher


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    from transformers import AutoTokenizer
    previous=ROOT.parent/'WSPL-executive-v2'
    old_manifest=previous/'experiments/source_pins/neural_executive_v2.manifest.json'
    expected_manifest='43dabbb5b31820586f368de2037e2673767c71eafa89c55833d2d1fa6b60f5fc'
    if digest(old_manifest)!=expected_manifest:raise ValueError('Historical manifest identity differs')
    old=json.loads(old_manifest.read_text())
    config=old['runner_config']
    blocks,specs=load_library(config['library'])
    library=library_document(blocks,specs)
    author=ROOT.parent/'WSPL-controller-author-v1/results/controller_author_v1/replies.jsonl'
    replies={r['id']:r for r in map(json.loads,author.read_text().splitlines()) if r['arm']=='trained'}
    migrated={}
    for kind,entry in config['library'].items():
        source=replies[entry['id']]['source'].strip()
        fence=re.fullmatch(r'```(?:wspl|text)?\s*\n([\s\S]*?)\n```',source)
        if fence:source=fence.group(1)
        migrated[kind]=compile_source('wspl 0.3;\n'+source).modules['Block'].sha256
        if migrated[kind]!=blocks[kind].sha256:raise ValueError('Module migration changed weights')
    tokenizer=AutoTokenizer.from_pretrained(old['constructor_kwargs']['model_path'],local_files_only=True)
    lengths={}
    result_dir=previous/'results/neural_executive_v2'
    old_result=json.loads((result_dir/'results.json').read_text())
    if old_result['status']!='complete' or old_result['manifest_sha256']!=expected_manifest:
        raise ValueError('Historical study is not the completed approved run')
    for split in ('train','development','test'):
        rows=[json.loads(x) for x in (result_dir/(split+'.jsonl')).read_text().splitlines()]
        maximum=0;completion_max=0;completion_total=0
        for start in range(0,len(rows),128):
            messages=[[dict(role='user',content=prompt(r['task'],library)),
                       dict(role='assistant',content=teacher(r['task'],library))] for r in rows[start:start+128]]
            full=tokenizer([tokenizer.apply_chat_template(m,tokenize=False) for m in messages],add_special_tokens=False)['input_ids']
            prefixes=tokenizer([tokenizer.apply_chat_template(m[:1],tokenize=False,add_generation_prompt=True) for m in messages],add_special_tokens=False)['input_ids']
            for ids,prefix in zip(full,prefixes):
                if ids[:len(prefix)]!=prefix:raise ValueError('Chat boundary changed')
                maximum=max(maximum,len(ids));completion_max=max(completion_max,len(ids)-len(prefix))
                completion_total+=len(ids)-len(prefix)
        if maximum>2048 or completion_max>512:raise ValueError('Context/generation cap insufficient')
        lengths[split]=dict(rows=len(rows),max_tokens=maximum,max_completion_tokens=completion_max,completion_tokens=completion_total)
    # Preserve every retained runtime/model binding, verifying against its old hash.
    resources={}
    for path,expected in config['resources'].items():
        if path==config['prior_adapter'] or path.endswith('EXECUTIVE_CURRICULUM_V2.md'):continue
        if digest(path)!=expected:raise ValueError('Previous resource changed: '+path)
        resources[path]=expected
    extras=[ROOT/'docs/wspl/UNIFIED_EXECUTIVE_V1.md',ROOT/'docs/wspl/UNIFICATION_V1.md',Path(sys.executable),
            result_dir/'replies.jsonl',result_dir/'test.jsonl',result_dir/'scores.json',result_dir/'results.json',
            author,result_dir/'train.jsonl',result_dir/'development.jsonl',old_manifest]
    for path in extras:resources[str(path.resolve())]=digest(path)
    report=dict(status='passed',module_identity_preserved=migrated,library_sha256=object_digest(library),
                token_audit=lengths,resources=len(resources),new_model_predictions=0,training_launched=False)
    (ROOT/'docs/wspl/unified_preparation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    spec=dict(model_id='unified-executive-v1',source_roots=old['source_roots'],
              constructor=old['constructor'],constructor_kwargs=old['constructor_kwargs'],runner=old['runner'],
              runner_config=dict(resources=resources,library=config['library'],language='wspl0.3',context=2048,
              epochs=2,accumulation=8,lr=.0002,warmup=160,wall_guard_seconds=43200,generation_tokens=512,
              previous_replies=str((result_dir/'replies.jsonl').resolve()),
              previous_tasks=str((result_dir/'test.jsonl').resolve()),previous_scores=str((result_dir/'scores.json').resolve()),
              previous_manifest_sha256='43dabbb5b31820586f368de2037e2673767c71eafa89c55833d2d1fa6b60f5fc'))
    path=ROOT/'experiments/source_pins/unified_executive_v1.spec.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(spec,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
