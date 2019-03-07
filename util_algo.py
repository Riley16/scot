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
            policy[s] = max([(mdp.reward(s) + gamma * sum([mdp.P[s, a, succ] * value_function_old[succ] * float(not mdp.is_terminal(s))
                                   for succ in range(nS)]), a) for a in range(nA)])[1]

            value_function[s] = mdp.reward(s) + gamma * sum([mdp.P[s, int(policy[s]), succ] * value_function_old[succ] * float(not mdp.is_terminal(s))
                                     for succ in range(nS)])

        eps = max(np.absolute(value_function - value_function_old))
    print('VI iterations to convergence: %d' % k)
    return value_function, policy

def monte_carlo(mdp, agent, T:int, n:int=1, eps:float=1e-2):
    """
    Learn value function of a policy by using n-th visit Monte Carlo sampling.

    Parameters:
    ----------
    P, nS, nA, gamma:
        define at beginning of file
    agent: agent object (see agent.py)
    T: maximum length of each episode
    n: n-th visit Monte Carlo, -1 for every-visit

    Returns:
    ----------
    value_function: np.ndarray[nS]
    """
    nS = mdp.nS
    nA = mdp.nS
    gamma = mdp.gamma
    
    N = np.zeros(nS)                    # track visits to each state
    G = np.zeros(nS, dtype=np.float32)  # track rewards for each state
    V_pi_old = np.zeros(nS)             # initialize value function
    iters = 0                           # track iterations

    assert T > 0                        # must be non-zero episode length

    # iterate until max(abs(V_pi_old - V_new)) < eps
    print('Running {}-th visit Monte Carlo policy evaluation...'.format(n))
    while True:
        # sample an episode
        traj = []
        mdp.reset()         # reset mdp for each episode
        s_i = mdp.start     # set start state for each episode
        for i in range(T):
            a_i = agent.get_action(s_i)
            s_j, r_i, is_done = mdp.step(s_i, a_i)
            traj.append((s_i, a_i, r_i))
            if is_done:
                break
            s_i = s_j

        # generate accumulated rewards at each time step
        G_t = np.zeros(nS)
        for i in range(G_t.shape[0]):
            for j, t in enumerate(i, traj):
                G_t[i] += (gamma ** j) * t[-1]

        # update N, G, and V_pi for each state
        V_pi_new = V_pi_old
        for t, g in zip(traj, G_t):
            s = t[0]
            if N[s] < n:
                N[s] += 1
                G[s] += g
                V_pi_new[s] = G[s] / N[s]

        # check for convergence
        if np.amax(np.abs(V_pi_new - V_pi_old)) < eps:
            break

        # prepare next rollout
        iters += 1
        V_pi_old = V_pi_new

    print('Monte Carlo iterations to converge: {}'.format(iters+1))

    return V_pi_new

# compute expected  linear feature counts mu[s][a] under optimal policy
# USE DETERMINISTIC POLICIES (AS INPUTS) FOR NOW
def get_feature_counts(mdp, policy, tol=1e-6):
    nS = mdp.nS
    nA = mdp.nA
    gamma = mdp.gamma

    # get policy-conditioned transition model
    P_pol = mdp.get_pol_trans(policy)

    # discounted expected feature counts under given policy for state s
    mu = np.zeros((nS, mdp.weights.shape[0]), dtype=float)
    # discounted expected feature counts mu_sa[s, a] after taking action a from state s
    # and then following the given policy
    mu_sa = np.zeros((nS, nA, mdp.weights.shape[0]), dtype=float)

    k = 0
    eps = tol + 1
    while eps > tol and k < 100:
        k = k + 1
        mu_old = np.copy(mu)
        for s in range(nS):
            mu[s] = mdp.s_features[s] + \
                    gamma * sum([P_pol[s, succ] * mu_old[succ] *
                                float(not mdp.is_terminal(s)) for succ in range(nS)])
            # mu[s] = mdp.s_features[s] + \
            #         gamma * sum([mdp.P[s, int(policy[s]), succ] * mu_old[succ] *
            #                     float(not mdp.is_terminal(s)) for succ in range(nS)])

        eps = np.max(np.absolute(mu - mu_old))

    for s in range(nS):
        for a in range(nA):
            mu_sa[s, a] = mdp.s_features[s] + \
                       gamma * sum([mdp.P[s, a, succ] * mu[succ] *
                                    float(not mdp.is_terminal(s)) for succ in range(nS)])

    print('Bellman iterations to convergence: %d' % k)

    return mu, mu_sa


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
