import numpy as np
import random


class Wrapper(object):
    def __init__(self, env, agent, log=True):
        ''' Initialize wrapper for running the env '''

        # initialize policy and env for evaluation
        self.env = env
        self.agent = agent
        self.log = log

    def eval_episodes(self, n_episodes, s_start=None, horizon=None):
        ''' Evaluate episodes with policy '''
        total_r = []
        trajectories = []
        for i in range(n_episodes):
            R, traj = self.eval_episode(s_start, horizon=horizon)

            total_r.append(R)
            trajectories.append(traj)
            if self.log:
                print("Agent log: {}".format(self.env.log))
                print("Agent trajectory: {}".format(self.env.traj))
        return total_r, trajectories

    def eval_episode(self, s_start=None, horizon=None):
        ''' Evaluate one episode between env and agent '''
        done = False
        total_r = 0
        s = self.env.reset(s_start=s_start)
        # finite horizon case
        if horizon is not None:
            t = 0
            while not done and t < horizon:
                t += 1
                a = self.agent.get_action(s)
                s, r, done = self.env.step(s, a)
                total_r += r
        # infinite horizon case
        else:
            while not done:
                a = self.agent.get_action(s)
                s, r, done = self.env.step(s, a)
                total_r += r
        return total_r, self.env.traj

    def sample(self, state):
        a = self.agent.get_action(state)
        s, r, done = self.env.step(s, a)
        return a, r, s, done
