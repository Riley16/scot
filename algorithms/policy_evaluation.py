''' Contains policy evaluation functions: Every/First visit Monte Carlo, Temporal Difference learning '''
import numpy as np


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


@rename('Every-visit Monte Carlo')
def every_visit_monte_carlo(wrapper, n_eps:int, eps_len:int):
    """
    Learn value function of a policy by using n-th visit Monte Carlo sampling.

    Parameters:
    ----------
    wrapper:        Wrapper object (see wrapper.py)
    T:              maximum length of each episode
    eps:            epsilon convergence to stop sampling

    Returns:
    ----------
    value_function: np.ndarray[nS]
    """

    nS = wrapper.env.nS
    gamma = wrapper.env.gamma

    N = np.zeros(nS)                    # track visits to each state
    G = np.zeros(nS, dtype=np.float32)  # track rewards for each state
    V_pi_old = np.zeros(nS)             # initialize value function

    assert eps_len > 0

    for _ in range(eps_len):
        # sample an episode
        _, traj = wrapper.eval_episodes(1, s_start=None, horizon=eps_len)
        traj = traj[0] # each step is (s, a, r, s')

        # add initial rewards to trajectory
        start_state = wrapper.env.start
        start_r = wrapper.env.reward(start_state)

        # accomodate for this by shifting trajectory rewards
        for i in range(1, len(traj)):
            _, _, r, _ = traj[i-1]
            s, _, _, s_ = traj[i]
            traj[i] = (s, None, r, s_)
        traj[0] = (start_state, None, start_r, traj[0][-1])
        last_state = traj[-1][-1]
        traj.append((last_state, None, wrapper.env.reward(last_state), None))

        # generate accumulated rewards at each time step
        G_t = np.zeros(len(traj))
        for i in range(G_t.shape[0]):
            for j, t in enumerate(traj[i:]):
                G_t[i] += (gamma ** j) * t[2]

        # update N, G, and V_pi for each state
        V_pi_new = np.copy(V_pi_old)
        for t, g in zip(traj, G_t):
            s = t[0]
            N[s] += 1
            G[s] += g
            V_pi_new[s] = G[s] / N[s]

        # prepare next rollout
        V_pi_old = np.copy(V_pi_new)

    return V_pi_new


@rename('First-visit Monte Carlo')
def first_visit_monte_carlo(wrapper, n_eps:int, eps_len:int):
    """
    Learn value function of a policy by using n-th visit Monte Carlo sampling.

    Parameters:
    ----------
    wrapper:        Wrapper object (see wrapper.py)
    T:              maximum length of each episode
    eps:            epsilon convergence to stop sampling

    Returns:
    ----------
    value_function: np.ndarray[nS]
    """

    nS = wrapper.env.nS
    gamma = wrapper.env.gamma

    N = np.zeros(nS)                    # track visits to each state
    G = np.zeros(nS, dtype=np.float32)  # track rewards for each state
    V_pi_old = np.zeros(nS)             # initialize value function

    assert eps_len > 0

    for _ in range(n_eps):
        # track visits to states
        first_visits = np.zeros((nS))

        # sample an episode
        _, traj = wrapper.eval_episodes(1, s_start=None, horizon=eps_len)
        traj = traj[0] # each step is (s, a, r, s')

        # add initial rewards to trajectory
        start_state = wrapper.env.start
        start_r = wrapper.env.reward(start_state)

        # accomodate for this by shifting trajectory rewards
        for i in range(1, len(traj)):
            _, _, r, _ = traj[i-1]
            s, _, _, s_ = traj[i]
            traj[i] = (s, None, r, s_)
        traj[0] = (start_state, None, start_r, traj[0][-1])
        last_state = traj[-1][-1]
        traj.append((last_state, None, wrapper.env.reward(last_state), None))

        # generate accumulated rewards at each time step
        G_t = np.zeros(len(traj))
        for i in range(G_t.shape[0]):
            for j, t in enumerate(traj[i:]):
                G_t[i] += (gamma ** j) * t[2]

        # update N, G, and V_pi for each state
        V_pi_new = np.copy(V_pi_old)
        for t, g in zip(traj, G_t):
            s = t[0]
            if first_visits[s] == 0:
                first_visits[s] += 1
                N[s] += 1
                G[s] += g
                V_pi_new[s] = G[s] / N[s]

        # prepare next rollout
        V_pi_old = np.copy(V_pi_new)

    print(n_eps)

    return V_pi_new


@rename('Temporal Difference learning')
def temporal_difference(wrapper, n_samp, step_size=0.1):
    '''
    Learn the value function for a given MDP environment and policy with Temporal Difference learning

    Parameters:
    ----------
    wrapper:    Wrapper object (see wrapper.py)
    step_size:  size of update
    reset:      whether to restart environment after reaching end state
    eps:        convergence error

    Returns:
    -------
    V_pi:   value function of the policy
    '''
    env = wrapper.env
    agent = wrapper.agent

    nS = env.nS
    gamma = env.gamma
    V_pi = np.zeros(nS, dtype=np.float32)
    env.reset()
    curr_state = env.start

    for _ in range(n_samp):
        # sample next tuple
        _, reward, next_state, done = wrapper.sample(curr_state)
        V_pi[curr_state] = V_pi[curr_state] + step_size * (
                env.reward(curr_state) + gamma * V_pi[next_state] - V_pi[curr_state])

        # reset environment if done
        if done:
            end_reward = env.reward(next_state)
            V_pi[next_state] = V_pi[next_state] + step_size * (end_reward - V_pi[next_state])
            curr_state = env.reset(s_start=None)
        else:
            curr_state = next_state

    return V_pi

if __name__ == '__main__':
    from tests import BrownNiekum
    from algorithms.value_iteration import value_iteration
    test = BrownNiekum()

    value_function_opt, policy = value_iteration(test.env)
    
    # change this to test other functions
    policy_eval_func = temporal_difference

    if policy_eval_func == temporal_difference:
        value_function_est = policy_eval_func(test.wrapper, **{'n_samp': 500, 'step_size':0.1})
    elif policy_eval_func == every_visit_monte_carlo:
        value_function_est = policy_eval_func(test.wrapper, **{'eps_len': 10, 'n_eps':50})
    elif policy_eval_func == first_visit_monte_carlo:
        value_function_est = policy_eval_func(test.wrapper, **{'eps_len': 10, 'n_eps':50})

    # compare value functions
    print('Optimal policy: {}'.format(policy))
    print('Optimal value function from Value Iteration: {}'.format(value_function_opt))
    print('Estimated value function from {}: {}'.format(policy_eval_func.__name__, value_function_est))
    