import numpy as np
from util_algo import *
from scipy.optimize import linprog
from agent import *
from wrapper import *
import time

# scipy.random
np.random.seed(2)

def SCOT(mdp, s_start, w, verbose=False):
    """
    Implements the Set Cover Optimal Teaching (SCOT) algorithm from
    "Machine Teaching for Inverse Reinforcement Learning:
    Algorithms and Applications", Brown and Niekum (2019)

    Args:
        mdp: MDP environment
        s_start: list of possible initial states
        w: weights of linear reward function of expert teacher agent
            (featurization computed by MDP environment) as a numpy array

    Returns:
        D: list of maximally informative machine teaching trajectories
            represented as lists of (s, a, r, s') experience tuples
    """

    t = time.time()

    # compute optimal policy pi_opt
    _, teacher_pol = value_iteration(mdp)  # using variation of VI code from HW1
    #print("Teacher policy: {}".format(teacher_pol))
    # convert teacher policy to stochastic policy
    teacher_pol = det2stoch_policy(teacher_pol, mdp.nS, mdp.nA)

    # compute expected feature counts mu[s][a] under optimal policy
    mu, mu_sa = get_feature_counts(mdp, teacher_pol)

    print("VI, feature counts")
    print(time.time() - t)
    t = time.time()

    # compute BEC of teacher as list of vectors defining halfspaces of linear reward
    # function parameters implied by teacher's policy
    BEC = np.empty((mdp.nS*mdp.nA, w.shape[0]))

    # get features for all MDP states
    # phi_s = np.array([list(mdp.s_features[s]) for s in range(mdp.nS)]).astype(float)

    # NOT NEEDED CURRENTLY
    # T_pi = mdp.get_pol_trans(teacher_pol)

    # compute BEC for teacher policy
    for a in range(mdp.nA):
        BEC[a*mdp.nS:(a+1)*mdp.nS] = mu - mu_sa[:, a]
        # FOR ANALYTICAL COMPUTATION BY NG REFERENCED IN BROWN AND NIEKUM (2019), CURRENTLY NOT WORKING, NOT NEEDED
        # pol_a = det2stoch_policy(np.full(mdp.nS, a), mdp.nS, mdp.nA)
        # T_a = mdp.get_pol_trans(pol_a)
        # # BEC[a*mdp.nS:(a+1)*mdp.nS] = (T_pi - T_a)@np.linalg.inv(np.eye(mdp.nS) - mdp.gamma*T_pi)@phi_s
        # test0 = T_pi - T_a
        # test1 = np.linalg.inv(np.eye(mdp.nS) - mdp.gamma*T_pi)
        # test2 = test0@test1
        # test = np.ones((mdp.nS, mdp.nS))@phi_s

    # remove trivial, duplicate, and redundant half-space constraints
    BEC = refineBEC(w, BEC)

    print("compute and refine teacher BEC")
    print(time.time() - t)
    t = time.time()


    print("BEC", BEC)
    # (1) compute candidate demonstration trajectories

    # STATISTICAL VALUES FOR NUMBER OF TRAJECTORIES WITH STOCHASTIC TRANSITIONS?
    m = 1

    teacher = Agent(teacher_pol, mdp.nS, mdp.nA)
    wrapper = Wrapper(mdp, teacher, False)

    demo_trajs = []

    # limit trajectory length to guarantee termination of algorithm
    # may want to increase max trajectory length for stochastic environments
    # may want to set trajectory limit to number of iterations required in VI/feature count computations
    H = mdp.nS  # /(1 + 1e-6 - mdp.noise)

    # FOR NOW USE ALL STATES,
    # LATER LIMIT TO JUST STATES WITH NON-ZERO START DISTRIBUTION PROBABILITIES
    for s in range(mdp.nS):
        demo_trajs += wrapper.eval_episodes(m, s, horizon=H)[1]

    if verbose:
        print("number of demonstration trajectories:")
        print(len(demo_trajs))


    print("get demonstration trajectories")
    print(time.time() - t)
    t = time.time()


    # print("demo_trajs", demo_trajs)
    # (2) greedy set cover algorithm to compute maximally informative trajectories
    U = set()
    for i in range(BEC.shape[0]):
        U.add(tuple(BEC[i].tolist()))
    D = []
    C = set()

    U_sub_C = U - C

    # greedy set cover algorithm
    """
        the set cover problem is to identify the smallest sub-collection of S whose union equals the universe.
        For example, consider the universe U={1,2,3,4,5} and the collection of sets S={{1,2,3},{2,4},{3,4},{4,5}}}
        """
    # SHOULD BE ABLE TO ELIMINATE ONE OF THE SET SUBTRACTIONS U - C
    # MIGHT BE THAT CANDIDATE TRAJECTORY SET DOESN'T COVER TEACHER BEC
    # CHECK THE CODE THAT i ELIMINATED IN MY LAST COMMIT, MIGHT HAVE BEEN A MISTAKE
    # SHOULD ALWAYS TERMINATE... AND SHOULDN'T ITERATE FOR MORE THAN THE NUMBER OF DEMONSTRATION TRAJECTORIES
    # SHOULDN'T BE RANDOM

    BECs_trajs = []
    for traj in demo_trajs:
        BECs_trajs.append(compute_traj_BEC(traj, mu, mu_sa, mdp, w))

    while len(U_sub_C) > 0:
        # while len(U - C) > 0:
        t_iter = time.time()
        t_list = []  # collects the cardinality of the intersection between BEC(traj|pi*) and U \ C
        BEC_list = []
        t_traj_BEC = time.time()
        for BEC_traj in BECs_trajs:
            BEC_list.append(BEC_traj)
            t_list.append(len(BEC_traj.intersection(U_sub_C)))
        print("demo traj BEC computation")
        print(time.time() - t_traj_BEC)
        t_greedy_index = t_list.index(max(t_list))
        t_greedy = demo_trajs[t_greedy_index]  # argmax over t_list to find greedy traj
        del BECs_trajs[t_greedy_index]
        del demo_trajs[t_greedy_index]
        D.append(t_greedy)
        C = C.union(BEC_list[t_greedy_index])
        U_sub_C = U - C

        # [8, 4, 4, 5, 25, 6, 5, 7, 5, 4]

        print("greedy set cover iteration")
        print(time.time() - t_iter)
        # t_iter = time.time()


    # while len(U_sub_C) > 0:
    #     # while len(U - C) > 0:
    #     t_iter = time.time()
    #     t_list = []  # collects the cardinality of the intersection between BEC(traj|pi*) and U \ C
    #     BEC_list = []
    #     t_traj_BEC = time.time()
    #     for traj in demo_trajs:
    #         BEC_traj = compute_traj_BEC(traj, mu, mu_sa, mdp, w)
    #         BEC_list.append(BEC_traj)
    #         # BEC_traj = BEC_traj.intersection(U - C)
    #         BEC_traj = BEC_traj.intersection(U_sub_C)
    #         t_list.append(len(BEC_traj))
    #     print("demo traj BEC computation")
    #     print(time.time() - t_traj_BEC)
    #     t_greedy_index = t_list.index(max(t_list))
    #     t_greedy = demo_trajs[t_greedy_index]  # argmax over t_list to find greedy traj
    #     del demo_trajs[t_greedy_index]
    #     D.append(t_greedy)
    #     C = C.union(BEC_list[t_greedy_index])
    #     U_sub_C = U - C
    #
    #     # [8, 4, 4, 5, 25, 6, 5, 7, 5, 4]
    #
    #     print("greedy set cover iteration")
    #     print(time.time() - t_iter)
    #     # t_iter = time.time()

    print("trajectories", D)
    lens = [len(s) for s in D]
    print(len(D), lens)
    return D


def compute_traj_BEC(traj, mu, mu_sa, mdp, w):
    # compute BEC of trajectory as numpy array
    BEC_traj_np = np.empty((mdp.nA * len(traj), w.shape[0]), dtype=float)
    for i in range(len(traj)):
        (s, a, r, s_new) = traj[i]
        for b in range(mdp.nA):
            BEC_traj_np[i * mdp.nA + b] = mu[s] - mu_sa[s, b]

    # normalize and remove trivial and redundant constraints from BEC of trajectory
    t_refine = time.time()
    BEC_traj_np = refineBEC(w, BEC_traj_np)
    print("refineBEC time")
    print(time.time()-t_refine)

    # convert BEC of trajectory to a set
    BEC_traj = set()
    for i in range(BEC_traj_np.shape[0]):
        BEC_traj.add(tuple(BEC_traj_np[i].tolist()))

    return BEC_traj


def removeLinRedundancies(BEC, bounds):
    b = np.zeros(BEC.shape[0])
    for i in range(BEC.shape[0] - 1, -1, -1):
        A = np.delete(BEC, i, 0)
        if A.shape[0] > 0:
            res = linprog(-BEC[i], A_ub=A, b_ub=b[:A.shape[0]], bounds=bounds)
            # res = linprog(-BEC[i], A_ub=A, b_ub=b[:A.shape[0]], bounds=bounds, options={"tol": 1e-06})
            if res.fun <= 0 and not res.status:
                BEC = A
    return BEC


def refineBEC(w, BEC):
    # remove trivial (all zero) constraints
    t_refine_start = time.time()
    triv_i = []
    for i in range(BEC.shape[0] - 1, -1, -1):
        if all(BEC[i] == np.zeros(w.shape[0])):
            triv_i.append(i)
    BEC = np.delete(BEC, triv_i, 0)

    # normalize BEC constraints
    for i in range(BEC.shape[0]):
        BEC[i] = BEC[i] / np.linalg.norm(BEC[i])

    # remove duplicate BEC constraints
    triv_i = []
    for i in range(BEC.shape[0]-1):
        for j in range(i+1, BEC.shape[0]):
            if np.array_equal(BEC[i], BEC[j]):
                triv_i.append(j)
    BEC = np.delete(BEC, triv_i, 0)

    print("refine without linaer redundancies")
    print(time.time() - t_refine_start)

    # remove redundant half-space constraints with linear programming
    bounds = tuple([(None, None) for _ in range(w.shape[0])])
    # TRY ADDING VARIABLE BOUNDS WHICH DO NOT AFFECT THE SOLUTIONS
    # bounds = tuple([(-1, 1) for _ in range(w.shape[0])])

    # add BEC constraints incrementally to try and improve performance
    # BEC_incr = np.zeros(BEC.shape)
    # for i in range(BEC.shape[0] - 1, -1, -1):
    #     BEC_incr[i] = BEC[i]
    #     BEC_temp = removeLinRedundancies(BEC_incr[i:BEC_incr.shape[0]], bounds)

    BEC = removeLinRedundancies(BEC, bounds)

    return BEC
