import numpy as np
from util_algo import *
from scipy.optimize import linprog

def SCOT(mdp, s_start, w):
    # implements the Set Cover Optimal Teaching (SCOT) algorithm from
    # "Machine Teaching for Inverse Reinforcement Learning:
    # Algorithms and Applications", Brown and Niekum (2019)

    #   Args:
    #       mdp:    MDP environment
    #       s_start:    list of possible initial states
    #       w:      weights of linear reward function of expert teacher agent (featurization computed by MDP environment) as a numpy array

    #   Returns:
    #       D:      list of maximally informative machine teaching trajectories
    #               represented as lists of (s, a, r, s') experience tuples

    # compute optimal behavioral equivalence class (BEC) of optimal policy pi_opt under R
    # compute optimal policy pi_opt
    _, teacher_pol = value_iteration(mdp)
    print(teacher_pol)
    mu, mu_sa = get_feature_counts(mdp, teacher_pol)
    teacher_pol = det2stoch_policy(teacher_pol, mdp.nS, mdp.nA)

    # NOT NEEDED FOR NOW, USE ALTERNATIVE ANALYTICAL MATRIX COMPUTATION FROM APPENDIX B, BROWN AND NIEKUM (2019)
    # compute expected feature counts mu[s][a] under optimal policy


    # compute BEC of teacher as list of vectors defining halfspaces of linear reward
    # function parameters implied by teacher's policy
    BEC = np.empty((mdp.nS*mdp.nA, w.shape[0]))

    # get features for all MDP states
    phi_s = np.array([list(mdp.s_features[s]) for s in range(mdp.nS)]).astype(float)

    T_pi = mdp.get_pol_trans(teacher_pol)
    for a in range(mdp.nA):
        pol_a = det2stoch_policy(np.full(mdp.nS, a), mdp.nS, mdp.nA)
        T_a = mdp.get_pol_trans(pol_a)
        # BEC[a*mdp.nS:(a+1)*mdp.nS] = (T_pi - T_a)@np.linalg.inv(np.eye(mdp.nS) - mdp.gamma*T_pi)@phi_s
        BEC[a*mdp.nS:(a+1)*mdp.nS] = mu - mu_sa[:, a]
        # test0 = T_pi - T_a
        # test1 = np.linalg.inv(np.eye(mdp.nS) - mdp.gamma*T_pi)
        # test2 = test0@test1
        # test = np.ones((mdp.nS, mdp.nS))@phi_s

    # remove all trivial (all zero) constraints
    triv_i = []
    for i in range(BEC.shape[0] - 1, -1, -1):
        if all(BEC[i] == np.zeros(w.shape[0])):
            triv_i.append(i)
    BEC = np.delete(BEC, triv_i, 0)

    # normalize BEC constraints
    for i in range(BEC.shape[0]):
        BEC[i] = BEC[i]/np.linalg.norm(BEC[i])

    # remove duplicate BEC constraints
    triv_i = set()
    for i in range(BEC.shape[0]):
        for j in range(BEC.shape[0]-1, i, -1):
            if all(BEC[i] == BEC[j]):
                triv_i.add(j)
    BEC = np.delete(BEC, list(triv_i), 0)

    # tmp = np.copy(BEC[0:2])
    # BEC[0:2] = np.copy(BEC[2:4])
    # BEC[2:4] = tmp
    print(BEC)

    # remove redundant half-space constraints with linear programming
    bounds = tuple([(None, None) for _ in range(w.shape[0])])
    for i in range(BEC.shape[0] - 1, -1, -1):
        c = -BEC[i]
        A = np.delete(BEC, i, 0)
        b = np.zeros(A.shape[0])
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, options={"disp": True})
        # DOES THIS MAKE SENSE? DOES AN UNCONSTRAINED PROBLEM NECESSARILY INDICATE A PROBLEM, OR JUST THAT TOO MANY CONSTRAINTS HAVE BEEN REMOVED?
        # if res.status:
        #     print("Removal of linear redundancies in teacher BEC unsuccessful.")
        #     exit()
        if res.fun <= 0 and not res.status:
            BEC = A

    print(BEC)

    D = BEC

    # compute candidate demonstration trajectories


    # greedy set cover algorithm to compute maximally informative trajectories



    return D


