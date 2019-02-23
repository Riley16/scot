import numpy as np


# Bellman backup operator for a (deterministic?) policy pi
def bellman(states, est, p, f, pi):
    # Args:
    #   est: quantities being estimated with Bellman backup as a list
    pass


# Value Iteration, returns optimal value function and implied optimal policy
def value_iteration(mdp, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """
    nS = mdp.nS
    nA = mdp.nA
    gamma = mdp.gamma
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    k = 0
    eps = tol + 1
    while eps > tol and k < 100:
        k = k + 1
        value_function_old = np.copy(value_function)

        # NEED TO ACCOUNT FOR TERMINAL STATES? NIEKUM IS INCLUDING TERMINAL AND INITIAL REWARDS
        for s in range(nS):
            policy[s] = max([(mdp.reward(s) + sum([mdp.P[s, a, succ] * gamma * value_function_old[succ] * float(not mdp.is_terminal(s))
                                   for succ in range(nS)]), a) for a in range(nA)])[1]

            value_function[s] = mdp.reward(s) + sum([mdp.P[s, int(policy[s]), succ] * gamma * value_function_old[succ] * float(not mdp.is_terminal(s))
                                     for succ in range(nS)])
            # policy[s] = max([(sum([mdp.P[s, a, succ] * (mdp.reward(succ) + gamma * value_function_old[succ]) * float(not mdp.is_terminal(s))
            #                        for succ in range(nS)]), a) for a in range(nA)])[1]
            #
            # value_function[s] = sum([mdp.P[s, int(policy[s]), succ] * (mdp.reward(succ) + gamma * value_function_old[succ]) * float(not mdp.is_terminal(s))
            #                          for succ in range(nS)])

        eps = max(np.absolute(value_function - value_function_old))
    print('VI iterations to convergence: %d' % k)
    print('value function:')
    print(value_function)
    print('policy')
    print(policy)

    # print(P)
    print('')

    return value_function, policy


# compute expected feature counts mu[s][a] under optimal policy
def get_feature_counts():
    pass

# equivalent BEC formulation, Ng and Russell, 2000
def get_trans_mat(MDP, pol):
    pass

# remove redundant half-space constraints with linear programming


# greedy set cover algorithm


def det2stoch_policy(det_pol, nS, nA):
    stoch_pol = np.zeros((nS, nA))
    for s in range(nS):
        stoch_pol[s, int(det_pol[s])] = 1.0
    return stoch_pol


def stoch2det_policy(stoch_pol, nS):
    det_pol = np.zeros(nS)
    for s in range(nS):
        det_pol[s] = stoch_pol[s].argmax()
    return det_pol
