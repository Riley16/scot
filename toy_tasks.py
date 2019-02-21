import numpy as np
import random
from env import Grid
from typing import List

Rewards = [1, -1] # for w_white, w_gray

# Task 1: Cooridor will be one long strip of positively rewarded tiles and a terminal state at the end
cooridor_len = 15
cooridor = np.zeros((3, cooridor_len))
cooridor[0,:] = 1
cooridor[2,:] = 1
cooridor[1, -1] = 2 # terminal state
print(cooridor)

# Task 2:
grid_w = 10
grid_h = 12
zig_zag = np.ones((grid_h, grid_w))
left_end = True
for row in range(0, grid_h, 2):
    zig_zag[row, :] = 0
    if left_end: # put stopper at end to form the maze
        zig_zag[row + 1, 0] = 0
        left_end = False
    else:
        zig_zag[row + 1, -1] = 0
        left_end = True
zig_zag[-1, -1] = 2 # terminal state
print(zig_zag)
