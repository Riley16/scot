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
        self.env = env # Grid(height, width, gamma, gray_sq, gray_r, white_r, term_r)
        self.agent = agent
        self.log = log

    def eval_episodes(self, n_episodes):
        ''' Evaluate episodes with policy '''
        total_r = []
        trajectories = []
        for _ in range(n_episodes):
            R, traj = self._eval_episode()
            total_r.append(R)
            trajectories.append(traj)
            if self.log:
                print("Agent log: {}".format(self.env.log))
                print("Agent trajectory: {}".format(self.env.traj))
        return total_r, trajectories

    #- internal functions -#
    def _eval_episode(self):
        ''' Evaluate one episode between env and agent '''
        done = False
        total_r = 0
        s = self.env.reset()
        while not done:
            a = self.agent.get_action(s)
            s, r, done = self.env.step(s, a)
            total_r += r
        return total_r, self.env.traj

