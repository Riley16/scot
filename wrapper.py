import numpy as np
import random
from env import Grid
from typing import List


class Wrapper(object):
    '''
    __func__ : default python function
    func     : external functionality
    _func    : internal functionality
    '''

    #- external functions -#
    def __init__(self, env, agent, log=True):
        ''' Initialize wrapper for running the env '''

        # initialize policy and env for evaluation
        self.env = env
        self.agent = agent
        self.log = log

    def eval_episodes(self, n_episodes, s_start=None):
        ''' Evaluate episodes with policy '''
        total_r = []
        trajectories = []
        for i in range(n_episodes):
            R, traj = self._eval_episode(s_start)

            total_r.append(R)
            trajectories.append(traj)
            if self.log:
                print("Agent log: {}".format(self.env.log))
                print("Agent trajectory: {}".format(self.env.traj))
        return total_r, trajectories

    #- internal functions -#
    def _eval_episode(self, s_start=None):
        ''' Evaluate one episode between env and agent '''
        done = False
        total_r = 0
        s = self.env.reset(s_start=s_start)
        while not done:
            a = self.agent.get_action(s)
            s, r, done = self.env.step(s, a)
            total_r += r
        return total_r, self.env.traj

