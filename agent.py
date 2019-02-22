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
        ''' Initialize wrapper for running the env '''

        if policy is None:
            policy = np.full([nS, nA], 1/nA)
        self.policy = policy

    def update_policy(self):
        ''' Updates policy '''
        raise NotImplementedError

    def get_action(self, s):
        return (np.cumsum(self.policy[s]) > np.random.random()).argmax()

