import numpy as np
from wrapper import Wrapper
from agent import Agent
from env import Grid
from algorithms.value_iteration import value_iteration
from util import det2stoch_policy


class Cooridor(object):
    def __init__(self):
        cooridor_width = 6
        cooridor_height = 3
        squares = []
        for i in range(cooridor_width):
            new_square = [int(cooridor_height / 2), i]
            squares.append(new_square)

        features_sq = [
            {
                'color': "grey",
                'reward': 0,
                'squares': squares
            },
            {
                'color': "red",
                'reward': 10,
                'squares': [[int(cooridor_height / 2), cooridor_width - 1]]
            }
        ]
        self.env  = Grid(cooridor_height, cooridor_width, gamma=0.9, white_r=-1, features_sq=features_sq,
                start_corner=False, end_pos=(int(cooridor_height/2), cooridor_width - 1))
        self.policy = self.init_policy()
        self.agent = Agent(self.policy, self.env.nS, self.env.nA)
        self.wrapper = Wrapper(self.env, self.agent, log=True)

    def init_policy(self):
        _, policy = value_iteration(self.env)
        print('Policy from VI: {}'.format(policy))
        return policy

class Loop(object):
    def __init__(self):
        grid_w = 4
        grid_h = 4
        loop_w = 2
        loop_h = 2
        vert_start = int((grid_h - loop_h) / 2)
        vert_end = int(((grid_h - loop_h) / 2) + loop_h)
        horz_start = int((grid_w - loop_w) / 2)
        horz_end = int(((grid_w - loop_w) / 2) + loop_w)

        squares = []
        for row in range(vert_start, vert_end + 1):
            for col in (horz_start, horz_end):
                new_square = [row, col]
                squares.append(new_square)
        for row in (vert_start, vert_end):
            for col in range(horz_start, horz_end + 1):
                new_square = [row, col]
                squares.append(new_square)
        features_sq = [
            {
                'color': "grey",
                'reward': -1,
                'squares': squares
            },
            {
                'color': "red",
                'reward': 10,
                'squares': [[vert_end - 1, horz_end - 1]]
            }
        ]

        self.env = Grid(grid_h, grid_w, gamma=0.9, white_r=-2, features_sq=features_sq, start_corner=False,
                        end_pos=(vert_end - 1, horz_end - 1))
        self.policy = self.init_policy()
        self.agent = Agent(self.policy, self.env.nS, self.env.nA)
        self.wrapper = Wrapper(self.env, self.agent, log=True)

    def init_policy(self):
        _, policy = value_iteration(self.env)
        print('Policy from VI: {}'.format(policy))
        return policy



class BasicGrid(object):
    ''' Basic grid test environment. '''
    def __init__(self):
        features = [
            {
                'color': 'gray',
                'reward': -10.0,
                'squares': [[1, 1], [1, 2], [2, 1], [2, 2]]
            },
            {
                'color':'rainbow',
                'reward': 100.0,
                'squares': [[3, 3]]
            }]
        self.env = Grid(4, 4, 0.75, white_r=-1.0, features_sq=features, start_corner=True, noise=0.0, weights=None)
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


class FromPaper(object):
    ''' 9x9 Grid environment with one hot features. '''
    def __init__(self):
        
        self.env = Grid(9, 9, 0.9, white_r=-1, gen_features="random", noise=0.0, n_features=8, weights="random", start_corner=True, one_hot=True)
        self.policy = self.init_policy()
        self.agent = Agent(self.policy, self.env.nS, self.env.nA)
        self.wrapper = Wrapper(self.env, self.agent, log=True)

    def init_policy(self):
        _, policy = value_iteration(self.env)
        print('Policy from VI: {}'.format(policy))
        return policy


class Random(object):
    ''' 9x9 Grid environment with one hot features. '''

    def __init__(self):
        self.env = Grid(3, 3, 0.9, white_r=-1, gen_features="random", noise=0.0, n_features=2, weights="random",
                        start_corner=False)
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
    Does not terminate for cases of
      0           0  (no motivation to terminate)
      1           1  (no motivation to terminate)
    '''
    def __init__(self):
        features = [
            {
                'color': 'gray',
                'reward': -10.0,
                'squares': [[1, 1]]
            }]
        # self.env = Grid(2, 3, 0.9, white_r=-1, features_sq=features, noise=0.0, start_corner=False)

        # sanity checks for various grid environment setup methods with basic Brown and Niekum environment

        # tests of explicit arbitrary feature inputs on a state-by-state basis
        # self.env = Grid(2, 3, 0.9, gen_features=[[1, 0],[1, 0],[1, 0],[1, 0],[0, 1],[1, 0]], weights=np.array([-1, -10]),
        #                 noise=0.0, start_corner=False)

        # tests of explicit arbitrary feature inputs on a row-column coordinate basis
        # self.env = Grid(2, 3, 0.9, gen_features=[[[1, 0],[1, 0],[1, 0]],[[1, 0],[0, 1],[1, 0]]], weights=np.array([-1, -10]),
        #                 noise=0.0, start_corner=False)

        # random reward weight assignment:
        # self.env = Grid(2, 3, 0.9, gen_features=[[[1, 0],[1, 0],[1, 0]],[[1, 0],[0, 1],[0, 0]]], n_features=2, weights="random",
        #                 noise=0.0, start_corner=False)

        # random feature assignments for known reward weights
        # self.env = Grid(2, 3, 0.9, gen_features="random", n_features=4, weights=np.array([-1, -10, 0, 0]),
        #                 noise=0.0, start_corner=False)

        # random feature assignments for ten features, random reward weights
        # self.env = Grid(2, 2, 0.9, gen_features="random", n_features=15, weights="random",
        #                 noise=0.0, start_corner=False)

        # basic test
        # self.env = Grid(2, 3, 0.9, gen_features=[[1, 0],[1, 0],[1, 0],[1, 0],[0, 1],[1, 0]], weights=np.array([-1, -10]),
        #                 noise=0.0, start_corner=False)
        #
        # tests of explicit arbitrary feature inputs on a state-by-state basis
        self.env = Grid(3, 3, 0.9, gen_features="random", n_features=3, weights="random",
                        noise=0.0, start_corner=False, verbose=False)

        self.policy = self.init_policy()
        self.agent = Agent(self.policy, self.env.nS, self.env.nA)
        self.wrapper = Wrapper(self.env, self.agent, log=True)

    def init_policy(self):
        _, policy = value_iteration(self.env)
        # print('Policy from VI: {}'.format(policy))
        policy = det2stoch_policy(policy, self.env.nS, self.env.nA)
        return policy
