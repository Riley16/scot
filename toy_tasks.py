import numpy as np
import SCOT as scot
from env import Grid
from typing import List


gamma = 0.9

# Task 1: Cooridor will be one long strip of positively rewarded tiles and a terminal state at the end
cooridor_width = 6
cooridor_height = 3
squares = []
for i in range(cooridor_width):
    new_square = [int(cooridor_height/2), i]
    squares.append(new_square)
print(squares)


features_sq = [
    {
        'color': "grey",
        'reward': -1,
        'squares': squares #[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                   # [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9]]
    },
    {
        'color': "red",
        'reward': 10,
        'squares': [[int(cooridor_height/2), cooridor_width - 1]] #[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                   # [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9]]
    }
]


cooridor = Grid(cooridor_height, cooridor_width, gamma, white_r=0, features_sq=features_sq,
                start_corner=False, end_pos=(int(cooridor_height/2), cooridor_width - 1))

output_cooridor = scot.SCOT(cooridor, None, cooridor.weights)
print(output_cooridor)
end_state = cooridor.grid_to_state((int(cooridor_height/2), cooridor_width - 2))
ground_truth = [(cooridor.grid_to_state((int(cooridor_height/2), cooridor_width - 2)), 3,
                 cooridor.reward(end_state), end_state)]


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
print(squares)
exit(0)
features_sq = [
    {
        'color': "grey",
        'reward': -1,
        'squares': squares
    },
    {
        'color': "red",
        'reward': 10,
        'squares': [[int(cooridor_height/2), cooridor_width - 1]] #[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                   # [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8], [2, 9]]
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

