import numpy as np
from env import Grid


class Agent(object):
    '''
    __func__ : default python function
    func     : external functionality
    _func    : internal functionality
    '''

    #- external functions -#
    def __init__(self, policy=None, nS=None, nA=None):
        # initialize agent policy to uniformly random if no initial policy is specified
        self.nS = nS
        self.nA = nA
        if policy is None:
            policy = np.full([nS, nA], 1/nA)
        self.policy = policy

    def set_policy(self, policy):
        ''' Updates policy '''
        self.policy = policy

    def get_action(self, s):
        return np.cumsum(self.policy[s] > np.random.random()).argmax()

