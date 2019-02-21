import numpy as np
import util_algo as util

def SCOT(mdp, s_start, R):
    # implements the Set Cover Optimal Teaching (SCOT) algorithm from
    # "Machine Teaching for Inverse Reinforcement Learning:
    # Algorithms and Applications", Brown and Niekum (2019)

    #   Args:
    #       mdp:    MDP environment
    #       s_start:    list of possible initial states
    #       R:      reward function of expert teacher agent

    #   Returns:
    #       D:      list of maximally informative machine teaching trajectories
    #               represented as lists of (s, a, r, s') experience tuples

    # compute optimal behavioral equivalence class (BEC) of optimal policy pi_opt under R

    # compute optimal policy pi_opt

    # compute expected feature counts mu[s][a] under optimal policy

    # compute BEC of teacher as list of vectors defining halfspaces of linear reward
    # function parameters implied by teacher's policy

    # normalize BEC vectors


    # remove redundant half-space constraints with linear programming

    # compute candidate demonstration trajectories


    # greedy set cover algorithm to compute maximally informative trajectories



    return D


