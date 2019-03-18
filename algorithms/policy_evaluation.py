''' Contains policy evaluation functions: Every/First visit Monte Carlo, Temporal Difference learning '''
import numpy as np

def every_visit_monte_carlo(wrapper, T:int=1, eps:float=1e-2):
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
    iters = 0                           # track iterations

    assert T > 0

    # iterate until epsilon convergence
    iters = 1
    while True:
        # sample an episode
        _, traj = wrapper.eval_episodes(1, s_start=None, horizon=T)
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
    
        # check for convergence
        if np.max(np.abs(V_pi_new - V_pi_old)) < eps:
            break

        # prepare next rollout
        iters += 1
        V_pi_old = np.copy(V_pi_new)

    return V_pi_new, iters

def first_visit_monte_carlo(wrapper, T:int=1, eps:float=1e-2):
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
    iters = 0                           # track iterations

    assert T > 0

    # iterate until epsilon convergence
    iters = 1
    while True:
        # track visits to states
        first_visits = np.zeros((nS))

        # sample an episode
        _, traj = wrapper.eval_episodes(1, s_start=None, horizon=T)
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
    
        # check for convergence
        if np.max(np.abs(V_pi_new - V_pi_old)) < eps:
            break

        # prepare next rollout
        iters += 1
        V_pi_old = np.copy(V_pi_new)

    return V_pi_new, iters

def temporal_difference(wrapper, step_size=0.1, reset=True, eps=1e-3):
    '''
    Learn optimal value function given an MDP environment with Temporal Difference learning

    Parameters:
    ----------
    wrapper:    Wrapper object (see wrapper.py)
    step_size:  size of update
    reset:      whether to restart environment after reaching end state
    eps:        convergence error

    Returns:
    -------
    V_pi:   optimal value function
    '''
    env = wrapper.env
    agent = wrapper.agent

    nS = env.nS
    gamma = env.gamma
    V_pi = np.zeros(nS, dtype=np.float32)
    curr_state = env.start

    # iterate until epsilon convergence
    iters = 1
    while True:
        # sample next tuplle
        action, reward, next_state, done = wrapper.sample(curr_state)
        new_val = V_pi[curr_state] + step_size * (reward + gamma * V_pi[next_state] - V_pi[curr_state])

        # determine whether to break or reset to keep going
        if done and not reset:
            break
        elif done and reset:
            env.reset(s_start=None)
        
        # epsilon convergence
        if np.abs(new_val - V_pi[curr_state]) < eps:
            break

        # prepare for next iteration
        V_pi[curr_state] = new_val
        curr_state = next_state
        iters += 1

    return V_pi, iters

if __name__ == '__main__':
    from tests import BrownNiekum
    from algorithms.value_iteration import value_iteration
    test = BrownNiekum()

    value_function_opt, policy = value_iteration(test.env)
    
    # value_function_est, _ = every_visit_monte_carlo(test.wrapper, T=10, eps=1e-5)
    # value_function_est, _ = first_visit_monte_carlo(test.wrapper, T=10, eps=1e-5)
    value_function_est, _ = temporal_difference(test.wrapper, step_size=0.1, reset=True, eps=1e-5)

    # compare value functions
    print('Optimal policy: {}'.format(policy))
    print('Value function from VI: {}'.format(value_function_opt))
    print('Value function from MC: {}'.format(value_function_est))
    