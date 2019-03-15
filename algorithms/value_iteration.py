''' Contains function to run value iteration '''
import numpy as np

def value_iteration(mdp, policy=None, r_weights=None, tol=1e-3, verbose=False):
    ''' Learn value function and policy by using value iteration method for a given
    gamma and mdpironment.

    Parameters:
    ----------
    mdp:        Grid object (see env.py)
    policy:     initial policy
    r_weights:  reward function weights
    tol:        convergence between iterations
    verbose:    verbose mode

    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    '''
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

        # Niekum includes terminal and initial state rewards
        for s in range(nS):
            if not eval_pol:
                policy[s] = max([(mdp.reward(s, w=r_weights) + gamma * sum([mdp.P[s, a, succ] * value_function_old[succ]
                                       for succ in range(nS)]), a) for a in range(nA)])[1]

            value_function[s] = mdp.reward(s, w=r_weights) + gamma * sum([mdp.P[s, int(policy[s]), succ] * value_function_old[succ] * float(not mdp.is_terminal(s))
                                     for succ in range(nS)])

        eps = max(np.absolute(value_function - value_function_old))

    if verbose:
        print('VI iterations to convergence: %d' % k)
        
    return value_function, policy