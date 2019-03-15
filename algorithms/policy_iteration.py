''' Contains functions for policy iteration and policy improvement. '''
import numpy as np
from wrapper import Wrapper
from tests import BrownNiekum
from algorithms.policy_evaluation import first_visit_monte_carlo, \
    every_visit_monte_carlo, temporal_difference

def policy_improvement(env, value_from_policy, old_policy):
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
    nS = env.nS
    nA = env.nA
    P = env.P

    for s in range(nS):
        # calculate Q value for taking all actions to see if currently policy is correct
        q_values = []
        for a in range(nA):
            action_value = 0
            for prob_of_trans in P[s][a]:
                # get next state
                grid_position = env.state_to_grid(s)
                grid_move = env.actions_to_grid[a]
                new_grid_position = grid_position + grid_move
                next_s = env.grid_to_state(new_grid_position)
                # get reward
                reward = env.reward(next_s)
                terminal = env.is_terminal(next_s)
                action_value += (prob_of_trans * (reward + gamma * value_from_policy[next_s]))
            q_values.append(action_value)
        new_policy[s] = q_values.index(max(q_values)) # find the action corresponding to largest Q

    return new_policy


def policy_iteration(env, agent, policy_eval_func, kwargs={}):
    '''Runs policy iteration for deterministic policy.

    Parameters:
    ----------
    env:                Grid object (see env.py)
    agent:              Agent object (see agent.py)
    policy_eval_func:   function for policy evaluation
    kwargs:             dictionary of optional arguments for policy_eval_func

    first_visit_monte_carlo: kwargs={'T':, 'eps':}
    every_visit_monte_carlo: kwargs={'T':, 'eps':}
    temporal_difference: kwargs={'step_size':, 'reset':, 'eps':}

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
    wrapper.agent.set_policy(old_policy)

    iters = 1
    while True:
        value_function, _ = policy_eval_func(wrapper, **kwargs)
        new_policy = policy_improvement(P, nS, nA, value_function, old_policy, gamma)

        # check to end policy iteration
        if new_policy - old_policy == 0:
            break
        
        # prepare for next iteration
        old_policy = np.copy(new_policy)
        wrapper.agent.set_policy(old_policy)
        iters += 1

    print('Policy iteration converged in {} iterations.'.format(i))

    return value_function, new_policy, i

if __name__ == '__main__':
    test = BrownNiekum()

    value_function, policy, i = policy_iteration(
        test.env, test.agent, first_visit_monte_carlo, kwargs={'T': None, 'eps': 1e-3})
    # value_function, policy, i = policy_iteration(
    #     test.env, test.agent, every_visit_monte_carlo, kwargs={'T': None, 'eps':1e-3})
    # value_function, policy, i = policy_iteration(
    #     test.env, test.agent, temporal_difference, kwargs={'step_size': 0.1, 'reset': True, 'eps':1e-3})
    print('Policy iteration converged in {} iterations.'.format(i))
