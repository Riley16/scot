import numpy as np


# Value Iteration, returns optimal value function and implied optimal policy
def value_iteration(mdp, policy=None, r_weights=None, tol=1e-3, verbose=False):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    CLEAN THIS UP
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
    if policy is None:
        policy = np.zeros(nS, dtype=int)
        eval_pol = False
    else:
        eval_pol = True

    k = 0
    eps = tol + 1
    while eps > tol and k < 100:
        k = k + 1
        value_function_old = np.copy(value_function)

        # NEED TO ACCOUNT FOR TERMINAL STATES? NIEKUM IS INCLUDING TERMINAL AND INITIAL REWARDS
        for s in range(nS):
            if not eval_pol:
                policy[s] = max([(mdp.reward(s, w=r_weights) + gamma * sum([mdp.P[s, a, succ] * value_function_old[succ] # * float(not mdp.is_terminal(s))
                                       for succ in range(nS)]), a) for a in range(nA)])[1]

            value_function[s] = mdp.reward(s) + gamma * sum([mdp.P[s, int(policy[s]), succ] * value_function_old[succ] * float(not mdp.is_terminal(s))
                                     for succ in range(nS)])

        eps = max(np.absolute(value_function - value_function_old))]

    if verbose:
        print('VI iterations to convergence: %d' % k)
    if not eval_pol:
        return value_function, policy
    else:
        return value_function


# compute expected  linear feature counts mu[s][a] under optimal policy
# USE DETERMINISTIC POLICIES (AS INPUTS) FOR NOW
def get_feature_counts(mdp, policy, tol=1e-6, verbose=False):
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

    if verbose:
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


def maxLikelihoodIRL(D, mdp, step_size = 0.01, eps=1e-02, max_steps=float("inf"), verbose=False):
    """
    Maximum Likelihood IRL: returns maximally likely reward function under Boltzmann policy for given set of rewards
    See Vroman 2011 (http://www.icml-2011.org/papers/478_icmlpaper.pdf) for original paper,
    Ratia 2012 (https://arxiv.org/pdf/1202.1558.pdf) for likelihood gradient formula.

    :param D: list of trajectories of an optimal policy in the given MDP mdp
    :param mdp: MDP environment
    :param eps: convergence criteria, float
    :param max_steps: max number of steps in gradient ascent, int
    :param verbose: verbosity of algorithmic reporting
    :return: r_weights: reward weights as np array
    """

    # initialize reward weights
    r_weights = np.random.rand(*mdp.weights.shape)
    print("Initial reward weights:")
    print(r_weights)

    # get state-action pairs observed in trajectories
    # (should technically be the only thing input to the algorithm, maybe move this code outside into utilizing code...
    sa_traj = []
    traj_states = []
    for traj in D:
        for i in range(len(traj)):
            sa_traj.append([traj[i][0], traj[i][1]])
            traj_states.append(traj[i][0])
    sa_traj = np.array(sa_traj)

    # convergence criteria
    iters = 0
    delta = eps + 1
    while delta > eps and iters < max_steps:
        iters += 1
        # compute value function and optimal policy under current reward estimate
        # Use max iterations of 100 in line with Vroman et al
        values, policy = value_iteration(mdp, r_weights=r_weights)
        policy = det2stoch_policy(policy, mdp.nS, mdp.nA)

        # compute Q-values
        Qvalues = np.array([[mdp.reward(s, r_weights) + mdp.gamma*np.sum([mdp.P[s, a, succ] * values[succ] for succ in range(mdp.nS)])
                             for a in range(mdp.nA)] for s in traj_states])

        # compute state-action likelihoods for all trajectory state-action pairs
        # temperature value for Boltzmann exploration policy (softmax with each exponent multiplied by beta)
        # used by Vroman et al
        beta = 0.5

        # apply Boltzmann temperature. subtract max Q-value so no large intermediate exponential values in likelihoods
        Qvalues_normalized = beta * (Qvalues - np.max(Qvalues))
        likelihoods = np.exp(Qvalues_normalized)
        for j in range(len(traj_states)):
            likelihood_sum = np.sum(likelihoods[j])
            likelihoods[j, :] /= likelihood_sum

        # compute state-action feature counts under current optimal policy
        mu, mu_sa = get_feature_counts(mdp, policy, tol=1.0e-03)

        # get likelihood gradient
        grad = np.sum(np.array([(mu_sa[s, a] - np.sum(np.array([likelihoods[s, b]*mu_sa[s, b] for b in range(mdp.nA)]), axis=0))
                                for s, a in sa_traj]), axis=0)

        # perform gradient ascent step, use step size schedule of gradient ascent iterations
        r_weights_old = np.copy(r_weights)
        r_weights += step_size/iters*grad

        # normalize r_weights to constrain L2 norm and force a unique reward weight vector
        r_weights /= np.linalg.norm(r_weights, ord=2)

        # convergence criterion
        delta = np.linalg.norm(r_weights - r_weights_old, 1)
        print(delta)

    if verbose:
        print("MLIRL iterations: {}".format(iters))
    return r_weights


