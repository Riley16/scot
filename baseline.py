import numpy as np
import random


def baseline(mdp, s_start):
    # implements the baseline algorithm to compare to SCOT.
    # The baseline feeds the learner random demonstrations,
    # representing the typical ML assumption of iid samples.

    #   Args:
    #       mdp:    MDP environment
    #       s_start:    list of possible initial states

    #   Returns:
    #       D:      list of random trajectories represented as
    #               a lists of (s, a, r, s') experience tuples

    max_demonstations = 50
    # choose random number of demonstrations
    num_demonstrations = random.randint(1, max_demonstations)

    # loop through and generate those random demonstrations
    D = []
    rand_s = -1
    for i in range(num_demonstrations):
        if rand_s < 0:
            rand_s = s_start[random.randint(len(s_start))]  # choose randomly among the possible start states
        else:
            rand_s = random.randint(mdp.nS)
        rand_a = random.randint(mdp.nA)
        grid_position = mdp.state_to_grid(rand_s)
        grid_move = mdp.actions_to_grid[rand_a]
        new_grid_position = grid_position + grid_move
        new_state = mdp.grid_to_state(new_grid_position)
        r = mdp.reward(new_state)
        D.append((rand_s, rand_a, r, new_state))

    return D
