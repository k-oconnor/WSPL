import unittest
from skillforge.codex_one_shot import audit_preview,settings,SUPERVISOR_INSTRUCTIONS


def preview(permission):
    def m(role,text):return dict(type='message',role=role,content=[dict(type='input_text',text=text)])
    return [dict(type='additional_tools',tools=[dict(name='functions',tools=[dict(name='request_user_input')])]),
            m('developer',SUPERVISOR_INSTRUCTIONS),m('developer','<permissions instructions>'+permission),
            m('developer','<collaboration_mode># Collaboration Mode: Default'),
            m('user','<environment_context>fixture'),m('user','task')]


class PermissionsTests(unittest.TestCase):
    def test_supervisor_profile_is_writable_with_no_native_shell(self):
        c=settings('catalog','instructions','supervisor')
        self.assertIn('sandbox_mode="danger-full-access"',c)
        self.assertIn('features.shell_tool=false',c)
        self.assertIn('features.unified_exec=false',c)
        self.assertIn('sandbox_mode="read-only"',settings('catalog','instructions'))
    def test_readonly_regression_fails_audit(self):
        with self.assertRaises(ValueError):audit_preview(preview('`sandbox_mode` is `read-only`: The sandbox only permits reading files.'),'task','supervisor')
        audit_preview(preview('`sandbox_mode` is `danger-full-access`: No filesystem sandboxing.'),'task','supervisor')
    def test_unknown_profile_rejected(self):
        with self.assertRaises(ValueError):settings('c','i','unexpected')
    def test_windows_instruction_newlines(self):
        p=preview('`sandbox_mode` is `danger-full-access`: No filesystem sandboxing.')
        p[1]['content'][0]['text']=SUPERVISOR_INSTRUCTIONS.replace('\n','\r\n')
        audit_preview(p,'task','supervisor')


if __name__=='__main__':unittest.main()
