''' Implements the maximum likelihood inverse reinforcement learning algorithm '''
import numpy as np
from algorithms.value_iteration import value_iteration
from util import det2stoch_policy, get_feature_counts

def max_likelihood_irl(D, mdp, step_size = 0.01, eps=1e-02, max_steps=float("inf"), verbose=False):
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

        print(mdp.nA, likelihoods.shape, mu_sa.shape)
        #print(sa_traj)
        # get likelihood gradient
        grad = beta*np.sum(np.array([(mu_sa[s, a] - np.sum(np.array([likelihoods[s, b]*mu_sa[s, b] for b in range(mdp.nA)]), axis=0))
                                for s, a in sa_traj]), axis=0)

        # perform gradient ascent step, use step size schedule of gradient ascent iterations
        r_weights_old = np.copy(r_weights)
        # L2_reg_factor = 0.001
        r_weights += step_size/iters*grad  #- L2_reg_factor * r_weights

        # normalize r_weights to constrain L2 norm and force a unique reward weight vector
        r_weights /= np.linalg.norm(r_weights, ord=2)

        # convergence criterion
        delta = np.linalg.norm(r_weights - r_weights_old, 1)
        if verbose:
            print(delta)

    if verbose:
        print("MLIRL iterations: {}".format(iters))
    return r_weights