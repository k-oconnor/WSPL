"""Starting policy example, copied into Astra's workspace, not a fixed solution."""
import json
import sys
sys.path.insert(0,'/inputs/code')
from skillforge.supervisor_workbench import StudentLab
from skillforge.student_control import prompt


class Policy:
    def __init__(self,root,declaration):
        self.student=StudentLab('/inputs/model','/inputs/parent')
    def reset(self,public):
        self.student.calls.clear()
    def act(self,observation):
        return self.student.generate(prompt(observation['observation'],observation['documentation']))
    def close(self):pass
