import numpy as np
from typing import Tuple, List, Dict
from actions import ACTIONS

# Grid environment
class Grid(object):
    '''
    __func__ : default python function
    func     : external functionality
    _func    : internal functionality
    '''

    def __init__(self, height:int, width:int, gamma:float, white_r:float,
                features_sq:List[Dict]=None, noise:float=0.0, weights=None, start_corner=True, start_dist=None):
        '''
        Initialize Grid environment

        Args:
            height: board height
            weight: board width
            gamma: discount factor
            features_sq: list of feature dictionaries in the following format:
                {
                    'color': str,
                    'reward': float,
                    'squares': List[List[int]]
                }
            noise: transition noise
            weights: reward function weights
            start_corner: whether to start from fixed location (or sample from start_dist)
            start_dist: distribution of start states, if start_corner=False
        
        '''

        #- Set metadata about MDP environment
        self.gamma = gamma
        self.nA = len(ACTIONS)
        self.nS = width*height
        self.actions_to_grid = {a: g for a, g in enumerate(ACTIONS)}
        self.grid_to_actions = {g: a for a, g in enumerate(ACTIONS)}

        #- Implement linear reward function weights in environment (associated with agent)
        if weights is None:
            self.weights = np.array([white_r] + [ft['reward'] for ft in features_sq], dtype=np.float32)
        else:
            assert isinstance(weights, np.ndarray)
            self.weights = weights
            
        #- Initialize features and board
        n_features = len(features_sq) + 1
        white_ft = tuple(1 if i == 0 else 0 for i in range(n_features))

        self.s_features = {s: white_ft for s in range(self.nS)} # initialize all features to white squares
        self.board = np.full([height, width], white_r, dtype=np.float32) # initialize all rewards to [white_r]

        #- Add additional features
        for idx, ft in enumerate(features_sq, 1):
            color = ft['color']
            reward = ft['reward']
            squares = ft['squares']
            assert color != None
            assert reward != None

            if not squares:
                n_color_sq = int(np.sqrt(width * height / n_features))
                squares = list(zip(
                    np.random.random_integers(0, width-1, n_color_sq), np.random.random_integers(0, height-1, n_color_sq)))
            print('{} squares: {}'.format(color, squares))

            ft_vec = tuple(1 if i == idx else 0 for i in range(n_features))
            for h, w in squares:
                self.board[h, w] = reward
                self.s_features[self.grid_to_state((h, w))] = ft_vec

        #- Transition matrix:
        # stochastic transitions following random step over any action with probability [noise] and following
        # deterministic transition function det_trans() with probability 1 - [noise]
        # in form of np array P[s, a, s'] giving the probability of transitioning from state s to s' after taking action a
        self.P = self._init_trans(noise, self.det_trans)

        #- Set special positions
        self.end = self.nS - 1

        #- Initialize start state: upper-left grid corner or from sample from start state distribution
        # uniformly sample over all states but terminal state if no distribution is input
        if start_corner is True:
            self.start = self.grid_to_state((0, 0))
            self.start_dist = None
        elif start_dist is None:
            self.start_dist = np.array([1/(self.nS-1) for _ in range(self.nS-1)])
            self.start = (np.cumsum(self.start_dist) > np.random.random()).argmax()
        else:
            self.start_dist = start_dist
            self.start = (np.cumsum(start_dist) > np.random.random()).argmax()

        #- Initialize agent attributes
        self.agent = self.start
        self.t = 0
        self.r = 0

        #- Set up logging
        self.log = [self.state_to_grid(self.start)]
        self.traj = []

    #- external functions -#
    def step(self, s:int, a: int):
        ''' Takes one step in the environment in response to action a '''
        successor = (np.cumsum(self.P[s, a]) > np.random.random()).argmax()

        r = self.reward(successor)
        self.log.append(self.state_to_grid(successor))
        self.r = self.gamma * self.r + r
        self.t += 1
        self.traj.append((s, a, r, successor))

        return successor, r, self.is_terminal(successor)

    def reward(self, s, w=None):
        if w is None:
            return np.dot(self.s_features[s], self.weights)
        return np.dot(self.s_features[s], w)

    def reset(self, s_start=None):
        ''' Reset environment to initial state of s_start if given, otherwise
        sample from start state distribution, otherwise to upper corner of (0, 0) '''
        self.t = 0
        self.r = 0

        if self.start_dist is None and s_start is None:
            self.start = self.grid_to_state((0, 0))
        elif s_start is not None:
            self.start = s_start
        else:
            self.start = (np.cumsum(self.start_dist) > np.random.random()).argmax()

        self.log = [self.state_to_grid(self.start)]
        self.traj = []
        return self.start

    def det_trans(self, s: int, a: int):
        ''' Takes one deterministic (noiseless or "slipless") step in the environment in response to action a '''
        new_pos = self._update_pos(self.state_to_grid(s), self.actions_to_grid[a])
        successor = self.grid_to_state(new_pos)
        # update position if action would not take the agent off the grid
        if new_pos[0] < self.board.shape[0] and new_pos[1] < self.board.shape[1] and new_pos[0] >= 0 and \
                new_pos[1] >= 0:
            # update agent position
            s = successor
        return s

    # computes transition matrix of environment conditioned on stochastic policy
    # stochastic policy input as numpy array policy[s, a] giving probability of taking action in state s
    # returns transition matrix conditioned on policy as numpy array P[s, s'] giving probability of transitioning to
    # state s' from state s under [policy]
    def get_pol_trans(self, policy):
        P_pol = np.zeros((self.nS, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                P_pol[s] += policy[s, a] * self.P[s, a]

        return P_pol

    def render(self):
        ''' Outputs state to console '''
        for h in range(self.board.shape[0]):
            s = [str(val) for val in self.board[h, :]]
            print('\t'.join(s))

    def state_to_grid(self, s):
        return s // self.board.shape[1], s % self.board.shape[1]

    def grid_to_state(self, g):
        return g[0] * self.board.shape[1] + g[1]

    def is_terminal(self, s):
        ''' Whether the agent has reached the terminal state '''
        return s == self.end #or self.t > 5

    @property
    def state(self):
        ''' Returns the state of the environment '''
        return self.board, self.agent, self.t, self.r

    #- internal functions -#
    def _update_pos(self, pos, action_grid):
        ''' Calculates updated position '''
        return pos[0] + action_grid[0], pos[1] + action_grid[1]

    def _init_trans(self, noise, det_trans):
        P = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            det_states = []
            # get all states that can be transitioned to from state s
            for a in range(self.nA):
                det_states.append(det_trans(s, a))
            for a in range(self.nA):
                P[s, a, det_trans(s, a)] += 1.0 - noise
                for slip_succ in det_states:
                    P[s, a, slip_succ] += noise / self.nA
        return P

