import numpy as np
from wrapper import Wrapper
from agent import Agent
from env import Grid
from actions import ACTIONS
from util_algo import *
from SCOT import SCOT

class BasicGrid(object):
    ''' Basic grid test environment. '''
    def __init__(self):
        features = [
            {
                'color': 'gray',
                'reward': 10.0,
                'squares': None
            }]
        self.env = Grid(4, 4, 0.75, white_r=1.0, features_sq=features, start_corner=True, noise=0.0, weights=None)
        self.policy = self.init_policy()
        self.agent = Agent(self.policy, self.env.nS, self.env.nA)
        self.wrapper = Wrapper(self.env, self.agent, log=True)

    def init_policy(self):
        ''' Agent moves left->right, up->down. '''
        policy = np.full(16, 2)
        policy[3] = 3
        policy[7] = 3
        policy[11] = 3
        policy[15] = 3
        print('Basic policy: {}'.format(policy))
        return policy


class MultipleFeatures(object):
    ''' Grid environment with multiple features. '''
    def __init__(self):
        features = [
            {
                'color': 'gray',
                'reward': -10.0,
                'squares': [[1, 1]]
            },
            {
                'color': 'fake_white',
                'reward': -10.0,
                'squares': [[1, 0]]
            }]
        self.env = Grid(2, 3, 0.9, white_r=-1, features_sq=features, noise=0.0, weights=None, start_corner=True)
        self.policy = self.init_policy()
        self.agent = Agent(self.policy, self.env.nS, self.env.nA)
        self.wrapper = Wrapper(self.env, self.agent, log=True)

    def init_policy(self):
        _, policy = value_iteration(self.env)
        print('Policy from VI: {}'.format(policy))
        return policy

class BrownNiekum(object):
    '''
    Brown and Niekum toy environment (2019)
    Our SCOT implementation currently handles cases of
      gray_r    white_r
      -10         -1
      -2          -1
      -1          -1
      -1          -2
    Does not handle cases of
      0           0  (may not be necessary)
      1           1  (may not be necessary)
    '''
    def __init__(self):
        features = [
            {
                'color': 'gray',
                'reward': -10.0,
                'squares': [[1, 1]]
            }]
        # self.env = Grid(2, 3, 0.9, white_r=-1, features_sq=features, noise=0.0, start_corner=False)

        # tests of arbitrary feature inputs using basic Brown and Niekum environment
        # self.env = Grid(2, 3, 0.9, gen_features=[[1, 0],[1, 0],[1, 0],[1, 0],[0, 1],[1, 0]], weights=np.array([-1, -10]),
        #                 noise=0.0, start_corner=False)
        # self.env = Grid(2, 3, 0.9, gen_features=[[[1, 0],[1, 0],[1, 0]],[[1, 0],[0, 1],[1, 0]]], weights=np.array([-1, -10]),
        #                 noise=0.0, start_corner=False)
        self.env = Grid(2, 3, 0.9, gen_features=[[[1, 0],[1, 0],[1, 0]],[[1, 0],[0, 1],[0, 0]]], n_features=2,
                        noise=0.0, start_corner=False)
        self.policy = self.init_policy()
        self.agent = Agent(self.policy, self.env.nS, self.env.nA)
        self.wrapper = Wrapper(self.env, self.agent, log=True)

    def init_policy(self):
        _, policy = value_iteration(self.env)
        print('Policy from VI: {}'.format(policy))
        return policy
