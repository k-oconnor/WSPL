import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from skillforge.codex_one_shot import audit_events, audit_preview, command, environment, collect


def events():
    return [dict(type='thread.started',thread_id='fresh'),dict(type='turn.started'),
            dict(type='item.completed',item=dict(type='reasoning',text='private summary')),
            dict(type='item.completed',item=dict(type='agent_message',text='```edits\nb3[:]=0\n```')),
            dict(type='turn.completed',usage=dict(input_tokens=100,cached_input_tokens=0,output_tokens=20))]


class CollectionTests(unittest.TestCase):
    def test_success_and_failure_audits(self):
        answer=events()[3]['item']['text']
        self.assertEqual(audit_events(events(),answer)['tool_calls'],0)
        for extra in [dict(type='error'),dict(type='turn.failed'),dict(type='turn.started'),
                      dict(type='item.completed',item=dict(type='command_execution'))]:
            with self.assertRaises(ValueError):audit_events(events()+[extra],answer)
        with self.assertRaises(ValueError):audit_events(events(),'wrong answer')
        broken=events();broken[-1]['usage']={}
        with self.assertRaises(ValueError):audit_events(broken,answer)

    def test_settings_and_environment(self):
        c=command('codex','catalog','instructions','empty','answer')
        self.assertIn('--ignore-user-config',c);self.assertIn('--ignore-rules',c)
        self.assertIn('--ephemeral',c);self.assertIn('forced_login_method="chatgpt"',c)
        self.assertIn('features.shell_tool=false',c)
        self.assertEqual(c[c.index('--model')+1],'gpt-6-astra')
        self.assertIn('suppress_unstable_features_warning=true',c)
        with patch.dict('os.environ',{'CODEX_THREAD_ID':'parent','OPENAI_API_KEY':'fake'}):
            self.assertNotIn('CODEX_THREAD_ID',environment());self.assertNotIn('OPENAI_API_KEY',environment())

    def test_fake_cli_collection(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fake=root/'fake.py';out=root/'out';answer=out/'answer.md'
            fake.write_text('import pathlib,json,sys\nsys.stdin.read()\n'+
                'pathlib.Path('+repr(str(answer))+').write_text('+repr(events()[3]['item']['text'])+')\n'+
                'for e in '+repr(events())+': print(json.dumps(e))\n')
            with patch('skillforge.codex_one_shot.preflight'), patch('skillforge.codex_one_shot.command',return_value=[sys.executable,str(fake)]):
                text,audit=collect('unused','catalog','instructions',out,'task',10,lambda:False)
            self.assertEqual(audit['thread_id'],'fresh');self.assertTrue(text.startswith('```edits'))
            self.assertTrue((out/'events.jsonl').exists())

    def test_fake_cli_timeout_and_no_retry(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            with patch('skillforge.codex_one_shot.preflight'), patch('skillforge.codex_one_shot.command',return_value=[sys.executable,'-c','import time;time.sleep(10)']):
                with self.assertRaisesRegex(RuntimeError,'call_wall_guard'):
                    collect('unused','catalog','instructions',Path(tmp)/'out','task',.05,lambda:False)

    def test_preview_rejects_tools_and_extra_context(self):
        def msg(role,text):return dict(type='message',role=role,content=[dict(type='input_text',text=text)])
        data=[dict(type='additional_tools',tools=[dict(name='functions',tools=[dict(name='request_user_input')])]),
              msg('developer',"Answer the user's request."),msg('developer','<permissions instructions>read only'),
              msg('developer','<collaboration_mode># Collaboration Mode: Default'),
              msg('user','<environment_context>empty'),msg('user','task')]
        audit_preview(data,'task')
        with self.assertRaises(ValueError):audit_preview(data+[msg('developer','reference answer')],'task')
        data[0]['tools'][0]['tools'].append(dict(name='shell'))
        with self.assertRaises(ValueError):audit_preview(data,'task')

    def test_streaming_startup_error_aborts_before_waiting_for_completion(self):
        import sys,time
        with tempfile.TemporaryDirectory() as tmp:
            code='import json,time;print(json.dumps({"type":"item.completed","item":{"type":"error","message":"startup warning"}}),flush=True);time.sleep(10)'
            started=time.monotonic()
            with patch('skillforge.codex_one_shot.preflight'),patch('skillforge.codex_one_shot.command',return_value=[sys.executable,'-c',code]):
                with self.assertRaisesRegex(ValueError,'error'):
                    collect('unused','catalog','instructions',Path(tmp)/'out','task',10,lambda:False)
            self.assertLess(time.monotonic()-started,5)
