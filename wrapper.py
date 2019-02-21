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
    def __init__(self, n_episodes:int, gamma:float, width:int, height:int,
        gray_r:float, white_r:float, term_r:float, gray_sq:List[List[int]]=[], policy=None, log=True):
        ''' Initialize wrapper for running the env '''

        # randomly initialize gray squares if not passed in
        if not gray_sq:
            n_gray_sq = int(np.sqrt(width * height))
            gray_sq = list(zip(
                np.random.random_integers(0, width-1, n_gray_sq), np.random.random_integers(0, height-1, n_gray_sq)))

        print('Gray squares: {}'. format(gray_sq))

        # initialize policy and env for evaluation
        self.env = Grid(height, width, gamma, gray_sq, gray_r, white_r, term_r)
        self.n_episodes = n_episodes
        self.log = log
        self.policy = self._init_policy(policy)

    def _init_policy(self, policy=None):
        ''' Initialize stochastic policy with uniform
            distribution over actions in all states '''
        if policy is None:
            policy = np.full([self.env.nS, self.env.nA], 1/self.env.nA)
        return policy

    def print_env(self):
        ''' Outputs state to console '''
        board, *_ = self.env.state
        for h in range(board.shape[0]):
            s = [str(val) for val in board[h, :]]
            print('\t'.join(s))

    def update_policy(self):
        ''' Updates policy '''
        raise NotImplementedError

    def eval_episodes(self):
        ''' Evaluate episodes with policy '''
        total_r = []
        for _ in range(self.n_episodes):
            self.env.reset()
            total_r.append(self._eval_episode())
            if self.log:
                print("Agent log: {}".format(self.env.log))
        return total_r

    #- internal functions -#
    def _eval_episode(self):
        ''' Evaluate one episode with policy '''
        done = False
        total_r = 0
        while not done:
            a = self.get_action(self.policy[self.env.agent, :])
            r, done = self.env.step(a)
            total_r += r
        return total_r

    def get_action(self, policy):
        return (np.cumsum(policy) > np.random.random()).argmax()
