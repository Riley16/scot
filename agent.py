''' Implements Agent class to interactive with Grid environment '''
import numpy as np


class Agent(object):
    def __init__(self, policy=None, nS=None, nA=None):
        ''' Initializes agent policy to uniformly random if no initial policy is specified '''
        self.nS = nS
        self.nA = nA
        if policy is None:
            policy = np.full([nS, nA], 1/nA)
        self.policy = policy

    def set_policy(self, policy):
        ''' Updates policy '''
        self.policy = policy

    def get_action(self, s):
        ''' Returns action from stochastic policy '''
        return np.cumsum(self.policy[s] > np.random.random()).argmax()
