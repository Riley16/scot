import numpy as np

class Grid(Object):
    '''
    __func__ : default python function
    func     : external functionality
    func_    : external attributes
    _func    : internal functionality
    '''

    def __init__(self, width:int, height:int, gamma:float, gray_sq:list[[int, int]],
        gray_r:float, white_r:float, term_r:float):
        ''' Initialize Grid environment '''
        # set metadata about mdp environment
        self.width = width
        self.height = height
        self.gamma = gamma

        # set up board
        self.board = np.full([width, height], white_r)
        for x, y in gray_sq:
            self.board[x, y] = gray_r
        self.board[-1, -1] = term_r

        # set special positions
        self.end = [-1, -1]
        self.start = [0, 0]
        self.agent = [0, 0]
        self.t = 0
        self.r = 0

    def step(self, action:list[int, int]):
        # assert legal action
        assert(sum(action) <= 1)
        if sum(action) > 0:
            # update agent position
            self.agent = self._update_agent(action)
            r = (self.gamma ** self.t) * self.board[self.agent[0], self.agent[1]]
        else:
            r = 0
        self.r += r
        self.t += 1

        return r, self._is_terminal()

    def _update_agent(self, action):
        return self.agent[0] + action[0], self.action[1] + action[1]

    def _is_terminal(self):
        return (self.agent[0] == self.end[0]) and (self.agent[1] == self.end[1])

    @property
    def get_actions_(self):
        actions = [
            [-1, -1], [0, -1], [1, -1],
            [-1, 0], [0, 0], [1, 0],
            [-1, 1], [0, 1], [1, 1],
        ]
        legal = []
        for action in actions:
            new_pos = self._update_agent(action)
            if new_pos[0] < width and new_pos[1] < height:
                legal.append(action)
        return legal

    @property
    def get_state_(self):
        return self.board, self.agent, self.t, self.r
