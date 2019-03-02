import numpy as np
import SCOT as scot
from env import Grid
from typing import List

Rewards = [1, -1] # for w_white, w_gray


gamma = 0.9


# Task 1: Cooridor will be one long strip of positively rewarded tiles and a terminal state at the end
cooridor_width = 6
cooridor_height = 3
#cooridor = np.zeros((cooridor_height, cooridor_width))
#cooridor[0,:] = 1
#cooridor[2,:] = 1
#cooridor[1, -1] = 2 # terminal state
#print(cooridor, "\n")
squares = []
for i in range(cooridor_width):
    new_square = [int(cooridor_height/2), i]
    squares.append(new_square)



features_sq = [
    {
        'color': "grey",
        'reward': -1,
        'squares': squares #[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                   # [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9]]
    }
]


cooridor = Grid(cooridor_height, cooridor_width, gamma, white_r=1, features_sq=features_sq, start_corner=False)

output = scot.SCOT(cooridor, None, cooridor.weights)
print(output)
exit(0)
ground_truth = []



#cooridor_soln = [(Grid.grid_to_state(1, cooridor_len - 2), 2,
 #                 Grid.reward(Grid.grid_to_state(1, cooridor_len - 1)), Grid.grid_to_state(1, cooridor_len - 1))]
#print(cooridor_soln)




# Task 2:
grid_w = 10
grid_h = 12
loop_w = 6
loop_h = 6
#loop = np.ones((grid_h, grid_w))
vert_start= int((grid_h - loop_h) / 2)
vert_end = int(((grid_h - loop_h) / 2) + loop_h)
horz_start = int((grid_w - loop_w) / 2)
horz_end = int(((grid_w - loop_w) / 2) + loop_w)
#loop[vert_start, horz_start:horz_end] = 0
#loop[vert_end, horz_start:horz_end] = 0
#loop[vert_start:vert_end + 1, horz_start] = 0
#loop[vert_start:vert_end + 1, horz_end] = 0
#loop[vert_end, horz_end - 1] = 2 # terminal state
#print(loop, "\n")

squares = []
for row in range(vert_start, vert_end):
    for col in range(horz_start, horz_end):
        new_square = [row, col]
        squares.append(new_square)

features_sq = [
    {
        'color': "grey",
        'reward': -1,
        'squares': squares
    }
]

loop = Grid(cooridor_height, cooridor_width, gamma, white_r=1, features_sq=features_sq, start_corner=False)

output = scot.SCOT(loop, None, cooridor.weights)
ground_truth = [(loop.grid_to_state())]


"""
# Task 3:
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
"""

