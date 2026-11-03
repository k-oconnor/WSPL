"""Bounded, auditable Codex collection. No scientific entry point here."""
import json
import os
from pathlib import Path
import subprocess
import time


DISABLED = ('code_mode code_mode_host shell_tool unified_exec unified_exec_tty '
    'shell_snapshot view_image image_generation multi_agent multi_agent_v2 apps plugins '
    'remote_plugin browser_use browser_use_external browser_use_full_cdp_access '
    'in_app_browser computer_use skill_search skill_mcp_dependency_install sleep_tool '
    'tool_suggest goals hooks workspace_dependencies in_app_chat in_app_local_automation '
    'fast_mode ultrafast_mode memories request_permissions_tool send_message_to_user_async').split()


def settings(catalog, instructions):
    return ['model="gpt-6-astra"', 'model_reasoning_effort="xhigh"',
        'approval_policy="never"', 'sandbox_mode="read-only"', 'web_search="disabled"',
        'forced_login_method="chatgpt"', 'model_provider="openai"',
        'project_doc_max_bytes=0', 'skills.include_instructions=false',
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


def command(cli, catalog, instructions, workspace, answer):
    args=[str(cli),'--no-daemon','exec','--strict-config','--ignore-user-config',
          '--ignore-rules','--ephemeral','--skip-git-repo-check','--json','--color','never',
          '-C',str(workspace),'--output-last-message',str(answer)]
    for setting in settings(catalog,instructions):args+=['-c',setting]
    return args+['-']


def audit_events(events, answer):
    """Reject tool use, extra turns, failures and unknown event variants."""
    starts=[];completed=[];threads=[];messages=[]
    for e in events:
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


def preflight(cli,catalog,instructions,folder,prompt):
    """Offline rendering only, with an empty auth/config home; no model call."""
    offline=folder/'offline-home';offline.mkdir()
    args=[str(cli),'--no-daemon','debug','prompt-input']
    for setting in settings(catalog,instructions):args+=['-c',setting]
    args+=[prompt]
    env=environment();env['CODEX_HOME']=str(offline.resolve())
    result=subprocess.run(args,cwd=folder/'empty',env=env,capture_output=True,encoding='utf-8',timeout=30)
    (folder/'prompt_preview.json').write_text(result.stdout,encoding='utf-8')
    (folder/'preflight.stderr').write_text(result.stderr,encoding='utf-8')
    if result.returncode or result.stderr.strip():raise ValueError('offline config preflight failed or warned')
    audit_preview(json.loads(result.stdout),prompt)


def audit_preview(messages,prompt):
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
    if (len(developers)!=3 or developers[0].strip()!="Answer the user's request."
        or not developers[1].startswith('<permissions instructions>')
        or not developers[2].startswith('<collaboration_mode># Collaboration Mode: Default')):
        raise ValueError('unexpected developer context')
    if len(users)!=2 or not users[0].startswith('<environment_context>') or users[1]!=prompt:
        raise ValueError('unexpected user context')


def collect(cli, catalog, instructions, folder, prompt, seconds, should_stop):
    folder=Path(folder);folder.mkdir(exist_ok=False)
    workspace=folder/'empty';workspace.mkdir()
    preflight(cli,catalog,instructions,folder,prompt)
    answer=folder/'answer.md'
    args=command(cli,catalog,instructions,workspace.resolve(),answer.resolve())
    (folder/'command.json').write_text(json.dumps(args,indent=2)+'\n',encoding='utf-8')
    (folder/'prompt.txt').write_text(prompt,encoding='utf-8')
    started=time.monotonic();reason=None
    with (folder/'events.jsonl').open('wb') as out, (folder/'stderr.log').open('wb') as err:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=out,stderr=err,env=environment())
        try:
            proc.stdin.write(prompt.encode('utf-8'));proc.stdin.close()
            while proc.poll() is None:
                if should_stop():reason='user_stop_or_study_guard';break
                if time.monotonic()-started>seconds:reason='call_wall_guard';break
                time.sleep(.2)
        finally:
            if proc.poll() is None:
                # Native CLI with daemon disabled; terminate process tree as well.
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
                else:proc.kill()
                proc.wait(timeout=15)
    if reason:raise RuntimeError(reason)
    if proc.returncode:raise RuntimeError('Codex exited '+str(proc.returncode)+'; see preserved stderr/events')
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    text=answer.read_text(encoding='utf-8')
    audit=audit_events(events,text)
    audit['active_seconds']=time.monotonic()-started
    (folder/'collection.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    return text,audit
