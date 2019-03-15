''' Implements some utility functions '''
from algorithms.value_iteration import value_iteration
import numpy as np
import random


def det2stoch_policy(det_pol, nS, nA):
    ''' Converts deterministic -> stochastic policy. '''
    stoch_pol = np.zeros((nS, nA))
    for s in range(nS):
        stoch_pol[s, int(det_pol[s])] = 1.0
    return stoch_pol


def stoch2det_policy(stoch_pol, nS):
    ''' Converts stochastic -> deterministic policy. '''
    det_pol = np.zeros(nS)
    for s in range(nS):
        det_pol[s] = stoch_pol[s].argmax()
    return det_pol


def get_feature_counts(env, policy, tol=1e-6, verbose=False):
    ''' Computes expected linear feature counts mu[s][a] under optimal policy.
    
    Parameters:
    ----------
    env:        Grid object (see env.py)
    policy:     deterministic policy
    tol:        change for convergence
    verbose:    print Bellman operations to converge

    Returns:
    mu          discounted expected feature counts under given policy for state s
    mu_sa       discounted expected feature counts mu_sa[s, a] after taking action a from state s
                and then following the given policy
    '''
    nS = env.nS
    nA = env.nA
    gamma = env.gamma

    # - Get policy-conditioned transition model
    P_pol = env.get_pol_trans(policy)

    # - Set up feature count returns
    mu = np.zeros((nS, env.weights.shape[0]), dtype=float)
    mu_sa = np.zeros((nS, nA, env.weights.shape[0]), dtype=float)

    # - Iterate until convergence
    k = 0
    eps = tol + 1
    while eps > tol and k < 100:
        k = k + 1
        mu_old = np.copy(mu)
        for s in range(nS):
            mu[s] = env.s_features[s] + \
                    gamma * sum([P_pol[s, succ] * mu_old[succ] *
                                float(not env.is_terminal(s)) for succ in range(nS)])
        eps = np.max(np.absolute(mu - mu_old))

    # - Accumulate rewards for mu_sa
    for s in range(nS):
        for a in range(nA):
            mu_sa[s, a] = env.s_features[s] + \
                       gamma * sum([env.P[s, a, succ] * mu[succ] *
                                    float(not env.is_terminal(s)) for succ in range(nS)])

    if verbose:
        print('Bellman iterations to convergence: %d' % k)
    return mu, mu_sa


def baseline(mdp, s_start, max_demonstrations=50):
    '''
    The baseline feeds the learner random demonstrations,
    representing the typical ML assumption of iid samples.

    Parameters:
    ----------
    mdp:        MDP environment
    s_start:    list of possible initial states

    Returns:
    ----------
    D:          list of random trajectories represented as
                a lists of (s, a, r, s') experience tuples
    '''
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
