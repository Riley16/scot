''' Contains functions for policy iteration and policy improvement. '''
import numpy as np
from wrapper import Wrapper
from algorithms.policy_evaluation import first_visit_monte_carlo, \
    every_visit_monte_carlo, temporal_difference
from util import det2stoch_policy

def policy_improvement(env, value_func, old_policy):
    '''Given the value function from policy improve the policy.

    Parameters
    ----------
    env:                Grid object (see env.py)
    value_function:     value function corresponding to old_policy
    old_policy:         policy to improve on

    Returns
    -------
    new_policy: np.ndarray[nS]
    '''

    new_policy = np.zeros_like(old_policy)
    gamma = env.gamma
    nS = env.nS
    nA = env.nA
    P = env.P

    #- Calculate argmax_a Q(s, a) for all s
    for s in range(nS):
        #- Accumulate Q(s, a) for all a
        q_values = np.zeros((nA))
        for a in range(nA):
            #- Compute expected reward, assuming stochastic env
            q_sa = value_func[s]
            for s_, p in enumerate(P[s, a]):
                reward = env.reward(s_)
                q_sa += p * (reward + gamma * value_func[s_])
            q_values[a] = q_sa
        #- Find action that maximizes Q value
        new_policy[s] = np.argmax(q_values)

    return new_policy


def policy_iteration(env, agent, policy_eval_func, kwargs={}):
    '''Runs policy iteration for deterministic policy.

    Parameters:
    ----------
    env:                Grid object (see env.py)
    agent:              Agent object (see agent.py)
    policy_eval_func:   function for policy evaluation
    kwargs:             dictionary of optional arguments for policy_eval_func

    first_visit_monte_carlo: kwargs={'n_eps':, 'eps_len':}
    every_visit_monte_carlo: kwargs={'n_eps':, 'eps_len':}
    temporal_difference: kwargs={'n_samp':, 'step_size':}

    Returns:
    ----------
    value_function:     np.ndarray[nS]
    policy:             np.ndarray[nS]
    '''
    nS = env.nS
    nA = env.nA

    value_function = np.zeros(nS)
    old_policy = np.zeros(nS, dtype=np.int32)
    new_policy = np.copy(old_policy)

    wrapper = Wrapper(env, agent, log=True)
    wrapper.agent.set_policy(det2stoch_policy(old_policy, nS, nA))

    iters = 1
    while True:
        value_function = policy_eval_func(wrapper, **kwargs)
        new_policy = policy_improvement(env, value_function, old_policy)

        # check to end policy iteration
        if np.amax(np.abs(new_policy - old_policy)) == 0:
            break
        
        # prepare for next iteration
        old_policy = np.copy(new_policy)
        wrapper.agent.set_policy(det2stoch_policy(old_policy, nS, nA))
        iters += 1

    print('Policy iteration converged in {} iterations.'.format(iters))
    print('Policy: {}'.format(new_policy))
    print('Value function: {}'.format(value_function))

    return value_function, new_policy

if __name__ == '__main__':
    from tests import BrownNiekum
    test = BrownNiekum()

    # value_function, policy = policy_iteration(
    #     test.env, test.agent, first_visit_monte_carlo, kwargs={'n_eps': 50, 'eps_len': 10})
    value_function, policy = policy_iteration(
        test.env, test.agent, every_visit_monte_carlo, kwargs={'n_eps': 50, 'eps_len': 10})
    # value_function, policy = policy_iteration(
    #     test.env, test.agent, temporal_difference, kwargs={'n_samp':1000', step_size': 0.1)
