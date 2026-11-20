import json
import unittest
import numpy as np
from skillforge.student_control import StudentExecutiveEpisode
from skillforge.factory_pilot import FactoryStudy


class ScriptedSession:
    def __init__(self, distances):
        self.distances=distances;self.steps=0;self.max_steps=100
    def observe(self):
        return {'nominal_tool_position_m':[self.distances[self.steps],0,0]}
    def act(self, command, ticks):
        assert ticks==1
        self.steps+=1


class ArrivalMarginTests(unittest.TestCase):
    def test_boundary_noise_does_not_end_reach_early(self):
        # Old 5 mm rule stopped on sample 3, although true sample 1
        # could be 5.056 mm. Conservative completion waits through sample 5.
        session=ScriptedSession([.006,.00495,.0042,.0035,.0029,.0024])
        episode=StudentExecutiveEpisode(session,{'motor':FactoryStudy().original},{'g':[0,0,0]})
        episode.execute(json.dumps({'op':'select','module':'motor'}))
        result=episode.execute(json.dumps({'op':'reach','goal':'g'}))
        self.assertEqual(result['status'],'observed_reached')
        self.assertEqual(result['ticks'],5)
        self.assertEqual(episode.observed_visits,['g'])

    def test_samples_outside_margin_never_claim_arrival(self):
        session=ScriptedSession([.0045]*11)
        episode=StudentExecutiveEpisode(session,{'motor':FactoryStudy().original},{'g':[0,0,0]})
        episode.execute(json.dumps({'op':'select','module':'motor'}))
        result=episode.execute(json.dumps({'op':'reach','goal':'g'}))
        self.assertEqual(result['status'],'chunk_ended')
        self.assertEqual(episode.observed_visits,[])


if __name__=='__main__':unittest.main()
