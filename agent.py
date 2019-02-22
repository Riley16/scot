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
        return (np.cumsum(self.policy[s]) > np.random.random()).argmax()

    def det2stoch_policy(self, det_pol):
        stoch_pol = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            print(det_pol[s])
            stoch_pol[s, int(det_pol[s])] = 1.0
        return stoch_pol

    def stoch2det_policy(self, stoch_pol):
        det_pol = np.zeros(self.nS)
        for s in range(self.nS):
            det_pol[s] = stoch_pol[s].argmax()
        return det_pol
