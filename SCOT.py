import numpy as np
from util_algo import *
from scipy.optimize import linprog
from agent import *
from wrapper import *


def refineBEC(w, BEC):
    # remove all trivial (all zero) constraints
    triv_i = []
    for i in range(BEC.shape[0] - 1, -1, -1):
        if all(BEC[i] == np.zeros(w.shape[0])):
            triv_i.append(i)
    BEC = np.delete(BEC, triv_i, 0)
    test=0
    # normalize BEC constraints
    for i in range(BEC.shape[0]):
        BEC[i] = BEC[i] / np.linalg.norm(BEC[i])

    # remove duplicate BEC constraints
    triv_i = set()
    for i in range(BEC.shape[0]):
        for j in range(BEC.shape[0] - 1, i, -1):
            if all(BEC[i] == BEC[j]):
                triv_i.add(j)
    BEC = np.delete(BEC, list(triv_i), 0)

    # remove redundant half-space constraints with linear programming
    bounds = tuple([(None, None) for _ in range(w.shape[0])])
    for i in range(BEC.shape[0] - 1, -1, -1):
        c = -BEC[i]
        A = np.delete(BEC, i, 0)
        b = np.zeros(A.shape[0])
        if A != [] and b != []:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds)
            if res.fun <= 0 and not res.status:
                BEC = A

    return BEC


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
        BEC[a*mdp.nS:(a+1)*mdp.nS] = mu - mu_sa[:, a]
        # FOR ANALYTICAL COMPUTATION BY NG REFERENCED IN BROWN AND NIEKUM (2019), CURRENTLY NOT WORKING, PROBABLY NOT NEEDED
        # pol_a = det2stoch_policy(np.full(mdp.nS, a), mdp.nS, mdp.nA)
        # T_a = mdp.get_pol_trans(pol_a)
        # # BEC[a*mdp.nS:(a+1)*mdp.nS] = (T_pi - T_a)@np.linalg.inv(np.eye(mdp.nS) - mdp.gamma*T_pi)@phi_s
        # test0 = T_pi - T_a
        # test1 = np.linalg.inv(np.eye(mdp.nS) - mdp.gamma*T_pi)
        # test2 = test0@test1
        # test = np.ones((mdp.nS, mdp.nS))@phi_s

    BEC = refineBEC(w, BEC)

    # (1) compute candidate demonstration trajectories

    # STATISTICAL VALUES FOR NUMBER OF TRAJECTORIES WITH STOCHASTIC TRANSITIONS?
    m = 1

    teacher = Agent(teacher_pol, mdp.nS, mdp.nA)
    wrapper = Wrapper(mdp, teacher, False)

    demo_trajs = []

    # FOR NOW USE ALL STATES,
    # LATER LIMIT TO JUST STATES WITH NON-ZERO START DISTRIBUTION PROBABILITIES
    for s in range(mdp.nS):
        demo_trajs += wrapper.eval_episodes(m, s)[1]

    # (2) greedy set cover algorithm to compute maximally informative trajectories
    U = set()
    for i in range(BEC.shape[0]):
        U.add(tuple(BEC[i].tolist()))
    D = []
    C = set()
    """
        the set cover problem is to identify the smallest sub-collection of S whose union equals the universe.
        For example, consider the universe U={1,2,3,4,5} and the collection of sets S={{1,2,3},{2,4},{3,4},{4,5}}}
        """
    while len(U - C) > 0:
        t_list = []  # collects the cardinality of the intersection between BEC(traj|pi*) and U \ C
        BEC_list = []
        for traj in demo_trajs:
            BEC_traj = compute_traj_BEC(traj, mu, mu_sa, mdp, w)
            BEC_list.append(BEC_traj)
            BEC_traj = BEC_traj.intersection(U - C)
            t_list.append(len(BEC_traj))
        t_greedy_index = t_list.index(max(t_list))
        t_greedy = demo_trajs[t_greedy_index]  # argmax over t_list to find greedy traj
        D.append(t_greedy)
        C = C.union(BEC_list[t_greedy_index])

    print(D)
    return D

def compute_traj_BEC(traj, mu, mu_sa, mdp, w):
    # compute BEC of trajectory as numpy array
    BEC_traj_np = np.empty((mdp.nA * len(traj), w.shape[0]), dtype=float)
    for i in range(len(traj)):
        (s, a, r, s_new) = traj[i]
        for b in range(mdp.nA):
            test = mu[s] - mu_sa[s, b]
            BEC_traj_np[i * mdp.nA + b] = mu[s] - mu_sa[s, b]

    # normalize and remove trival and redundant constraints from BEC of trajectory
    BEC_traj_np = refineBEC(w, BEC_traj_np)

    # convert BEC of trajectory to a set
    BEC_traj = set()
    for i in range(BEC_traj_np.shape[0]):
        BEC_traj.add(tuple(BEC_traj_np[i].tolist()))
    return BEC_traj

