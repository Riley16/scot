import numpy as np
from scipy.optimize import linprog
from agent import Agent
from wrapper import Wrapper
from algorithms.value_iteration import value_iteration
from util import det2stoch_policy, get_feature_counts



def scot(mdp, w, s_start=None, m=None, H=None, seed=None, verbose=False):
    """
    Implements the Set Cover Optimal Teaching (SCOT) algorithm from
    "Machine Teaching for Inverse Reinforcement Learning:
    Algorithms and Applications", Brown and Niekum (2019)

    Args:
        mdp: MDP environment
        s_start: list of possible initial states
        w (np.array): weights of linear reward function of expert teacher agent
            (featurization computed by MDP environment) as a numpy array
        m (int): number of sample demonstration trajectories to draw per start state
        H (int): horizon (max length) of demonstration trajectories
        verbose (Boolean): whether to print out algorithmic runtime information

    Returns:
        D: list of maximally informative machine teaching trajectories
            represented as lists of (s, a, r, s') experience tuples
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(2)

    # compute optimal policy pi_opt
    _, teacher_pol_det = value_iteration(mdp)  # using variation of VI code from HW1)
    # convert teacher policy to stochastic policy
    teacher_pol = det2stoch_policy(teacher_pol_det, mdp.nS, mdp.nA)

    # compute expected feature counts mu[s][a] under optimal policy
    mu, mu_sa = get_feature_counts(mdp, teacher_pol)

    # compute BEC of teacher as list of vectors defining halfspaces of linear reward
    # function parameters implied by teacher's policy
    BEC = np.empty((mdp.nS*mdp.nA, w.shape[0]))

    # compute BEC for teacher policy
    # for a in range(mdp.nA):
    #     BEC[a*mdp.nS:(a+1)*mdp.nS] = mu - mu_sa[:, a]
    BEC = np.empty((mdp.nS*(mdp.nA-1), w.shape[0]))
    i = 0
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            if a != teacher_pol_det[s]:
                BEC[i] = mu[s] - mu_sa[s, a]
                i += 1

    # remove trivial, duplicate, and redundant half-space constraints
    BEC = refineBEC(w, BEC)

    if verbose:
        print("BEC", BEC)

    # (1) compute candidate demonstration trajectories

    # number of demonstration trajectories to sample per start state
    if m is None:
        m = int(np.ceil(1/(1.0 - 0.95*mdp.noise)))

    teacher = Agent(teacher_pol, mdp.nS, mdp.nA)
    wrapper = Wrapper(mdp, teacher, False)

    demo_trajs = []

    # limit trajectory length to guarantee SCOT termination,
    # increase max trajectory length for stochastic environments
    # may want to set trajectory limit to number of iterations required in VI/feature count computations
    if H is None:
        H = mdp.nS

    # sample demonstration trajectories from each starting state (either from each state with non-zero probability mass
    # in the start state distribution of the MDP or a given set of start states
    # for s in range(mdp.nS):
    #     demo_trajs += wrapper.eval_episodes(m, s, horizon=H)[1]
    if s_start is None:
        for s in range(mdp.nS):
            if mdp.start_dist[s] > 0.0:
                demo_trajs += wrapper.eval_episodes(m, s, horizon=H)[1]
    else:
        for s in s_start:
            demo_trajs += wrapper.eval_episodes(m, s, horizon=H)[1]

    if verbose:
        print("number of demonstration trajectories:")
        print(len(demo_trajs))

    # (2) greedy set cover algorithm to compute maximally informative trajectories
    U = set()
    for i in range(BEC.shape[0]):
        U.add(tuple(BEC[i].tolist()))
    D = []
    C = set()

    U_sub_C = U - C
    if verbose:
        print("number of BEC constraints before set cover")
        print(len(U))

    # greedy set cover algorithm
    """
    the set cover problem is to identify the smallest sub-collection of S whose union equals the universe.
    For example, consider the universe U={1,2,3,4,5} and the collection of sets S={{1,2,3},{2,4},{3,4},{4,5}}}
    """
    BECs_trajs = []
    for traj in demo_trajs:
        BECs_trajs.append(compute_traj_BEC(traj, mu, mu_sa, mdp, w))

    while len(U_sub_C) > 0 and len(BECs_trajs) > 0:
        t_list = []  # collects the cardinality of the intersection between BEC(traj|pi*) and U \ C
        BEC_list = []
        for BEC_traj in BECs_trajs:
            BEC_list.append(BEC_traj)
            t_list.append(len(BEC_traj.intersection(U_sub_C)))
        t_greedy_index = t_list.index(max(t_list))
        t_greedy = demo_trajs[t_greedy_index]  # argmax over t_list to find greedy traj
        del BECs_trajs[t_greedy_index]
        del demo_trajs[t_greedy_index]
        D.append(t_greedy)
        C = C.union(BEC_list[t_greedy_index])
        U_sub_C = U - C
        if len(BECs_trajs) == 0:
            print("BEC_trajs empty")
            print(len(U_sub_C))
            print(U_sub_C)

    if verbose:
        print("trajectories", D)
        lens = [len(s) for s in D]
        print(len(D), lens)
    return D


def compute_traj_BEC(traj, mu, mu_sa, mdp, w):
    # compute BEC of trajectory as numpy array
    BEC_traj_np = np.zeros((mdp.nA * len(traj), w.shape[0]), dtype=float)
    for i in range(len(traj)):
        # COULD REMOVE REDUNDANT SA-PAIRS HERE WITH A SET OF (S, A, R, S_NEW) TUPLES
        (s, a, r, s_new) = traj[i]
        for b in range(mdp.nA):
            if b != a:
                BEC_traj_np[i * mdp.nA + b] = mu[s] - mu_sa[s, b]
            # if b == a:
            #     print(mu[s] - mu_sa[s, b])

    # normalize and remove trivial and redundant constraints from BEC of trajectory
    BEC_traj_np = refineBEC(w, BEC_traj_np)

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
            if res.fun <= 0 and not res.status:
                BEC = A
    return BEC


def refineBEC(w, BEC):
    # remove trivial (all zero) constraints
    triv_i = []
    z = np.zeros(w.shape[0])
    for i in range(BEC.shape[0]):
        if np.array_equal(BEC[i], z):
            triv_i.append(i)
    BEC = np.delete(BEC, triv_i, 0)

    # normalize BEC constraints
    for i in range(BEC.shape[0]):
        BEC[i] = BEC[i] / np.linalg.norm(BEC[i])

    # remove duplicate BEC constraints
    BEC_unique = set()
    for i in range(BEC.shape[0]):
        BEC_unique.add(tuple(BEC[i].tolist()))
    BEC = np.array(list(BEC_unique))

    # remove redundant half-space constraints with linear programming
    bounds = tuple([(None, None) for _ in range(w.shape[0])])
    BEC = removeLinRedundancies(BEC, bounds)

    return BEC
