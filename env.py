import numpy as np
from typing import Tuple, List
from actions import ACTIONS


class Grid(object):
    '''
    __func__ : default python function
    func     : external functionality
    _func    : internal functionality
    '''

    def __init__(self, width:int, height:int, gamma:float, gray_sq:List[List[int]],
        gray_r:float, white_r:float, term_r:float):
        ''' Initialize Grid environment '''
        # set metadata about mdp environment
        self.gamma = gamma
        self.n_actions = len(ACTIONS)
        self.actions_to_idx = {a: i for i, a in enumerate(ACTIONS)}
        self.idx_to_actions = {i: a for i, a in enumerate(ACTIONS)}

        # set up board
        self.board = np.full([width, height], white_r)
        for x, y in gray_sq:
            self.board[x, y] = gray_r
        self.board[-1, -1] = term_r

        # set special positions
        self.end = (width-1, height-1)
        self.start = (0, 0)
        self.agent = self.start
        self.t = 0
        self.r = 0

        # logging
        self.log = [self.agent]

    #- external functions -#
    def step(self, idx:int):
        ''' Makes one time step in the environment '''
        action = self.idx_to_actions[idx]
        if sum(action) > 0:
            # update agent position
            self.agent = self._update_pos(self.agent, action)
            r = (self.gamma ** self.t) * self.board[self.agent[0], self.agent[1]]
        else:
            r = 0

        self.log.append(self.agent)
        self.r += r
        self.t += 1

        return r, self._is_terminal()

    def reset(self):
        ''' Reset environment to initial state '''
        self.t = 0
        self.r = 0
        self.agent = [0, 0]
        self.log = [self.agent]

    def legal_actions(self, pos):
        ''' Returns all legal actions from current position '''
        legal = []
        for action in ACTIONS:
            new_pos = self._update_pos(pos, action)
            if new_pos[0] < self.board.shape[0] and new_pos[1] < self.board.shape[1] and new_pos[0] >= 0 and new_pos[1] >= 0:
                legal.append(self.actions_to_idx[action])
        return legal

    @property
    def state(self):
        ''' Returns the state of the environment '''
        return self.board, self.agent, self.t, self.r

    
    #- internal functions -#
    def _update_pos(self, pos, action):
        ''' Calculates updated position '''
        return pos[0] + action[0], pos[1] + action[1]

    def _is_terminal(self):
        ''' Whether the agent has reached the terminal state '''
        return (self.agent[0] == self.end[0]) and (self.agent[1] == self.end[1])


