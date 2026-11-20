"""Bounded, auditable Codex collection. No scientific entry point here."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import time
import threading


DISABLED = ('code_mode code_mode_host shell_tool unified_exec unified_exec_tty '
    'shell_snapshot view_image image_generation multi_agent multi_agent_v2 apps plugins '
    'remote_plugin browser_use browser_use_external browser_use_full_cdp_access '
    'in_app_browser computer_use skill_search skill_mcp_dependency_install sleep_tool '
    'tool_suggest goals hooks workspace_dependencies in_app_chat in_app_local_automation '
    'fast_mode ultrafast_mode memories request_permissions_tool send_message_to_user_async').split()


SUPERVISOR_INSTRUCTIONS = """Answer the user's request.
You are authorized to create, modify, save and reload files through the delegated
Python tool in its persistent Linux /work workspace, including trained weights,
LoRA adapters, datasets, programs and deployment artifacts. /inputs mounts are
read-only. The Python worker has no network and cannot access the host filesystem.
The collector's native shell tools are disabled; its permission display does not
restrict the separately delegated /work tool. Never infer a read-only /work policy
from historical logs: those are evidence from a superseded misconfigured run.
"""


def settings(catalog, instructions, profile='readonly'):
    if profile not in ('readonly','supervisor'):raise ValueError('Unknown collector profile')
    sandbox='danger-full-access' if profile=='supervisor' else 'read-only'
    return ['model="gpt-6-astra"', 'model_reasoning_effort="xhigh"',
        'approval_policy="never"', 'sandbox_mode='+json.dumps(sandbox), 'web_search="disabled"',
        'forced_login_method="chatgpt"', 'model_provider="openai"',
        'project_doc_max_bytes=0', 'skills.include_instructions=false',
        'suppress_unstable_features_warning=true',
        'skills.bundled.enabled=false', 'mcp_servers={}', 'agents.enabled=false',
        'features.skip_host_skill_discovery=true',
        'features.unbounded_connection_retries=false',
        'model_catalog_json='+json.dumps(Path(catalog).resolve().as_posix()),
        'model_instructions_file='+json.dumps(Path(instructions).resolve().as_posix())
        ] + ['features.'+f+'=false' for f in DISABLED]


def environment():
    # Do not inherit the parent agent's thread, API credentials or provider overrides.
    env={k:v for k,v in os.environ.items() if not k.startswith(('CODEX_','OPENAI_'))}
    env['CODEX_HOME']=str(Path.home()/'.codex')
    return env


def command(cli, catalog, instructions, workspace, answer, large_prompt=False,profile='readonly'):
    args=[str(cli),'--no-daemon','exec','--strict-config','--ignore-user-config',
          '--ignore-rules','--ephemeral','--skip-git-repo-check','--json','--color','never',
          '--model','gpt-6-astra',
          '-C',str(workspace),'--output-last-message',str(answer)]
    for setting in settings(catalog,instructions,profile):args+=['-c',setting]
    if large_prompt:args+=['-c','model_context_window=872000','-c','model_auto_compact_token_limit=800000']
    return args+['-']


def check_event(e):
    kind=e.get('type')
    if kind in ('thread.started','turn.started','turn.completed'):return
    if kind in ('item.started','item.updated','item.completed') and e.get('item',{}).get('type') in ('reasoning','agent_message'):return
    raise ValueError('tool, error or unknown event: '+str(kind)+'/'+str(e.get('item',{}).get('type')))


def audit_events(events, answer):
    """Reject tool use, extra turns, failures and unknown event variants."""
    starts=[];completed=[];threads=[];messages=[]
    for e in events:
        check_event(e)
        kind=e.get('type')
        if kind=='thread.started':threads.append(e['thread_id'])
        elif kind=='turn.started':starts.append(e)
        elif kind=='turn.completed':completed.append(e)
        elif kind in ('item.started','item.updated','item.completed'):
            item=e.get('item',{})
            if item.get('type') not in ('reasoning','agent_message'):
                raise ValueError('tool or unknown item: '+str(item.get('type')))
            if kind=='item.completed' and item.get('type')=='agent_message':messages.append(item['text'])
        else:raise ValueError('failure or unknown event: '+str(kind))
    if len(threads)!=1 or len(starts)!=1 or len(completed)!=1:
        raise ValueError('exactly one fresh thread and completed turn required')
    if not messages or messages[-1].strip()!=answer.strip():
        raise ValueError('saved answer differs from final agent message')
    usage=completed[0].get('usage',{})
    if any(type(usage.get(k)) is not int or usage[k]<0 for k in ('input_tokens','cached_input_tokens','output_tokens')):
        raise ValueError('missing usage accounting')
    return dict(thread_id=threads[0],usage=usage,agent_messages=len(messages),tool_calls=0)


def preflight(cli,catalog,instructions,folder,prompt,images=(),profile='readonly'):
    """Offline rendering only, with an empty auth/config home; no model call."""
    offline=folder/'offline-home';offline.mkdir()
    args=[str(cli),'--no-daemon','debug','prompt-input']
    for setting in settings(catalog,instructions,profile):args+=['-c',setting]
    # debug prompt-input has no file/stdin option. Large prompts use a digest
    # sentinel to audit context; actual collection still sends the full stdin.
    large=len(prompt.encode('utf-16-le'))>16000
    digest=hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    preview_prompt=('Full prompt supplied through exec stdin; SHA-256 '+digest) if large else prompt
    if large:args+=['-c','model_context_window=872000','-c','model_auto_compact_token_limit=800000']
    for path in images:args+=['--image',str(Path(path).resolve())]
    args+=['--',preview_prompt] if images else [preview_prompt]
    env=environment();env['CODEX_HOME']=str(offline.resolve())
    result=subprocess.run(args,cwd=folder/'empty',env=env,capture_output=True,encoding='utf-8',timeout=30)
    (folder/'prompt_preview.json').write_text(result.stdout,encoding='utf-8')
    (folder/'preflight.stderr').write_text(result.stderr,encoding='utf-8')
    if result.returncode or result.stderr.strip():raise ValueError('offline config preflight failed or warned')
    preview=json.loads(result.stdout)
    if images:
        audit_image_preview(preview,images)
    audit_preview(preview,preview_prompt,profile)
    (folder/'preflight_binding.json').write_text(json.dumps(dict(
        mode='context_only_digest_sentinel' if large else 'full_prompt',
        prompt_sha256=digest,prompt_utf8_bytes=len(prompt.encode('utf-8')),
        collection_transport='full UTF-8 stdin via codex exec -',
        limitation='Large-prompt preflight verifies configuration/context, not server-side prompt receipt.' if large else None),indent=2)+'\n',encoding='utf-8')


def audit_image_preview(preview,images):
    """Verify actual offline image bytes then normalize for the existing text audit."""
    import base64
    users=[m for m in preview if m.get('role')=='user']
    if len(users)!=2:raise ValueError('Unexpected image prompt roles')
    parts=users[-1]['content']
    if len(parts)!=3*len(images)+1:raise ValueError('Unexpected image framing')
    for i,path in enumerate(images):
        opening,image,closing=parts[i*3:i*3+3]
        if opening.get('type')!='input_text' or not opening.get('text','').startswith('<image name='):
            raise ValueError('Missing image opening')
        if closing!={'type':'input_text','text':'</image>'}:raise ValueError('Image closing')
        if image.get('type')!='input_image' or not image.get('image_url','').startswith('data:image/png;base64,'):
            raise ValueError('PNG image expected')
        if base64.b64decode(image['image_url'].split(',',1)[1],validate=True)!=Path(path).read_bytes():
            raise ValueError('Image bytes changed in prompt')
    users[-1]['content']=[parts[-1]]


def audit_preview(messages,prompt,profile='readonly'):
    tools=[];developers=[];users=[]
    for m in messages:
        if m.get('type')=='additional_tools':
            for namespace in m['tools']:
                if namespace.get('name')!='functions':raise ValueError('unexpected tool namespace')
                tools += [t['name'] for t in namespace['tools']]
        elif m.get('type')=='message':
            texts=[part['text'] for part in m.get('content',[]) if part.get('type')=='input_text']
            if m.get('role')=='developer':developers.extend(texts)
            elif m.get('role')=='user':users.extend(texts)
            else:raise ValueError('unexpected prompt role')
        else:raise ValueError('unexpected prompt item')
    if tools!=['request_user_input']:raise ValueError('unexpected tool exposure')
    expected=SUPERVISOR_INSTRUCTIONS.strip() if profile=='supervisor' else "Answer the user's request."
    if (len(developers)!=3 or developers[0].replace('\r\n','\n').strip()!=expected
        or not developers[1].startswith('<permissions instructions>')
        or not developers[2].startswith('<collaboration_mode># Collaboration Mode: Default')):
        raise ValueError('unexpected developer context')
    if profile=='supervisor':
        permissions=developers[1]
        if ('`sandbox_mode` is `danger-full-access`' not in permissions
            or 'The sandbox only permits reading files' in permissions
            or '`sandbox_mode` is `read-only`' in permissions):
            raise ValueError('Supervisor permission contract is not writable')
    if len(users)!=2 or not users[0].startswith('<environment_context>') or users[1]!=prompt:
        raise ValueError('unexpected user context')


def collect(cli, catalog, instructions, folder, prompt, seconds, should_stop, images=(),profile='readonly'):
    folder=Path(folder);folder.mkdir(exist_ok=False)
    workspace=folder/'empty';workspace.mkdir()
    archived=[]
    if len(images)>12:raise ValueError('At most 12 attached frames per response')
    for i,path in enumerate(images):
        from PIL import Image
        data=Path(path).read_bytes()
        if len(data)>4*1024*1024:raise ValueError('Image too large')
        with Image.open(path) as im:
            if im.format!='PNG' or im.size!=(256,256):raise ValueError('Expected 256x256 PNG frame')
            im.verify()
        copy=folder/f'input_{i:02d}.png';copy.write_bytes(data);archived.append(copy.resolve())
    preflight(cli,catalog,instructions,folder,prompt,archived,profile=profile)
    answer=folder/'answer.md'
    args=command(cli,catalog,instructions,workspace.resolve(),answer.resolve(),large_prompt=len(prompt.encode('utf-16-le'))>16000,profile=profile)
    if archived:
        args=args[:-1]
        for path in archived:args+=['--image',str(path)]
        args+=['--','-']
    (folder/'command.json').write_text(json.dumps(args,indent=2)+'\n',encoding='utf-8')
    (folder/'prompt.txt').write_text(prompt,encoding='utf-8')
    started=time.monotonic();reason=None
    offset=0;pending=b''
    with (folder/'events.jsonl').open('wb') as out, (folder/'stderr.log').open('wb') as err:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=out,stderr=err,env=environment())
        payload=prompt.encode('utf-8');delivery={}
        def send_prompt():
            try:
                delivery['bytes_written']=proc.stdin.write(payload)
                proc.stdin.close()
            except (OSError,ValueError) as exc:delivery['error']=str(exc)
        writer=threading.Thread(target=send_prompt,daemon=True);writer.start()
        try:
            while proc.poll() is None:
                if 'error' in delivery:raise RuntimeError('prompt stdin delivery failed: '+delivery['error'])
                with (folder/'events.jsonl').open('rb') as event_stream:
                    event_stream.seek(offset);chunk=event_stream.read();offset=event_stream.tell()
                pending+=chunk
                while b'\n' in pending:
                    line,pending=pending.split(b'\n',1)
                    if line.strip():check_event(json.loads(line))
                if should_stop():reason='user_stop_or_study_guard';break
                if time.monotonic()-started>seconds:reason='call_wall_guard';break
                time.sleep(.2)
        finally:
            if proc.poll() is None:
                # Native CLI with daemon disabled; terminate process tree as well.
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                else:proc.kill()
                proc.wait(timeout=15)
            writer.join(timeout=5)
    if reason:raise RuntimeError(reason)
    if proc.returncode:raise RuntimeError('Codex exited '+str(proc.returncode)+'; see preserved stderr/events')
    if writer.is_alive() or 'error' in delivery or delivery.get('bytes_written')!=len(payload):raise RuntimeError('incomplete prompt stdin delivery')
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    text=answer.read_text(encoding='utf-8')
    audit=audit_events(events,text)
    audit['active_seconds']=time.monotonic()-started
    audit['prompt_transport']=dict(bytes_written=delivery['bytes_written'],sha256=hashlib.sha256(payload).hexdigest(),transport='stdin',server_receipt_verified=False)
    audit['images']=[dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in archived]
    (folder/'collection.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    return text,audit
