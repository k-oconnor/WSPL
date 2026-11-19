import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from skillforge.factory_pilot import FactoryStudy, MOTOR_SOURCE, SELECT_SOURCE, request, packet
from skillforge.codex_one_shot import audit_image_preview, collect


class TestFactory(unittest.TestCase):
    def test_install_atomic(self):
        s=FactoryStudy();old=s.artifact()
        a=dict(probe=[.1,0,0],probe_ticks=3,motor=MOTOR_SOURCE,selector='invalid')
        with self.assertRaises(Exception):s.install(a)
        self.assertEqual(old,s.artifact())
        a['selector']=SELECT_SOURCE;s.install(a);self.assertEqual(s.probe_ticks,3)
    def test_request(self):
        self.assertEqual(request('{"tool":"finish","prediction":"done"}')['tool'],'finish')
        for a in [dict(tool='finish'),dict(tool='finish',prediction='',hidden=3),dict(tool='observe',prediction='x',bench='fault',cameras=[])]:
            with self.assertRaises(ValueError):request(json.dumps(a))
    def test_packet_scope(self):
        p=packet([],{});self.assertIn('No simulator object coordinates',p)
        self.assertIn('Astra is absent',p);self.assertIn('not deployment features',p)
    def test_image_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x.png';Image.new('RGB',(256,256)).save(p)
            image=dict(type='input_image',image_url='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())
            def preview():return [dict(role='user',content=[]),dict(role='user',content=[dict(type='input_text',text='<image name=test>'),dict(image),dict(type='input_text',text='</image>'),dict(type='input_text',text='prompt')])]
            x=preview();audit_image_preview(x,[p]);self.assertEqual(len(x[-1]['content']),1)
            image['image_url']='data:image/png;base64,ZmFrZQ=='
            with self.assertRaises(ValueError):audit_image_preview(preview(),[p])
    def test_runner_freeze_and_boundary(self):
        import run_factory_pilot as runner
        from test_factory_observation import Backend
        class Fake(Backend):
            def __init__(self,*args):super().__init__()
            def close(self):pass
        def rollout(study,folder,seed,mapping,offset,arm,cameras=()):
            Path(folder).mkdir()
            return dict(success=True,steps=3,goal_m=offset,final_error_m=.001,final_measured_error_m=.001,
                        final_telemetry={},diagnostic=None,frames=[],log=[])
        def collect(*args,**kwargs):
            return '{"tool":"finish","prediction":"retain"}',dict(usage={})
        with tempfile.TemporaryDirectory() as t:
            config=dict(resources={},wall_guard_seconds=30,cli='unused',catalog='unused',instructions='unused',changed_mapping=[[0,-1,0],[1,0,0],[0,0,1]],offsets=[[0,0,.02]])
            with patch.object(runner,'PandaBackend',Fake),patch.object(runner,'rollout',rollout),patch.object(runner,'collect',collect):
                runner.run(FactoryStudy,Path(t),config)
            result=json.loads((Path(t)/'results.json').read_text());self.assertEqual(result['status'],'complete')
            self.assertEqual(len(json.loads((Path(t)/'scores.json').read_text())),12)
            self.assertTrue((Path(t)/'artifact_final.json').exists())

    def test_image_collection_transport(self):
        import sys
        from test_codex_one_shot import events
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);image=root/'frame.png';Image.new('RGB',(256,256),(3,4,5)).save(image)
            out=root/'call';fake=root/'fake.py';answer=events()[3]['item']['text']
            fake.write_text('import sys,json,pathlib\nsys.stdin.read()\n'+
                'assert "--image" in sys.argv and sys.argv[-2:]==["--","-"]\n'+
                'assert pathlib.Path(sys.argv[sys.argv.index("--image")+1]).read_bytes()==pathlib.Path('+repr(str(image))+').read_bytes()\n'+
                'pathlib.Path('+repr(str(out/'answer.md'))+').write_text('+repr(answer)+')\n'+
                'for e in '+repr(events())+': print(json.dumps(e))\n')
            with patch('skillforge.codex_one_shot.preflight'),patch('skillforge.codex_one_shot.command',return_value=[sys.executable,str(fake),'-']):
                _,audit=collect('unused','catalog','instructions',out,'look',10,lambda:False,images=[image])
            self.assertEqual(len(audit['images']),1)
            self.assertEqual((out/'input_00.png').read_bytes(),image.read_bytes())


if __name__=='__main__':unittest.main()
