import numpy as np


# Bellman backup operator for a (deterministic?) policy pi
def bellman(states, est, p, f, pi):
    # Args:
    #   est: quantities being estimated with Bellman backup as a list
    pass


# Value Iteration, returns optimal value function and implied optimal policy
def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
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

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    k = 0
    eps = tol + 1
    while eps > tol:
        k = k + 1
        value_function_old = np.copy(value_function)

        for s in range(nS):
            policy[s] = max([(sum([prob * (r + gamma * value_function_old[new_state])
                                   for (prob, new_state, r, term) in P[s][a]]), a) for a in range(nA)])[1]

            value_function[s] = sum([prob * (r + gamma * value_function_old[new_state])
                                     for (prob, new_state, r, term) in P[s][policy[s]]])

        eps = max(np.absolute(value_function - value_function_old))
    print('VI iterations to convergence: %d' % k)
    print('value function:')
    print(value_function)
    print('policy')
    print(policy)

    # print(P)
    print('')

    ############################
    return value_function, policy


# compute expected feature counts mu[s][a] under optimal policy
def get_feature_counts():
    pass

# equivalent BEC formulation, Ng and Russell, 2000
def get_trans_mat(MDP, pol):
    pass

# remove redundant half-space constraints with linear programming


# greedy set cover algorithm



