import numpy as np

def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


@rename('Q-learning')
def q_learning(wrapper, n_samp, step_size=0.1, epsilon=0.1, horizon=None, traj_limit=100):
    '''
    Learn optimal value function given an MDP environment with Q-learning under a greedy epsilon policy

    Parameters:
    ----------
    wrapper:    Wrapper object (see wrapper.py)
    step_size:  size of update
    reset:      whether to restart environment after reaching end state
    eps:        convergence error

    Returns:
    -------
    V:      estimated optimal value function
    Q:      estimated optimal Q-function
    policy: estimated optimal policy
    '''
    env = wrapper.env
    agent = wrapper.agent

    if horizon is None:
        horizon = float("inf")
    t = 0

    nS = env.nS
    nA = env.nA
    gamma = env.gamma
    Q = np.zeros((nS, nA), dtype=np.float32)
    env.reset()
    curr_state = env.start
    num_trajs = 1
    for _ in range(n_samp):
        # sample next tuple
        t += 1
        if np.random.random() > epsilon:
            a = np.argmax(Q[curr_state])
        else:
            a = np.random.randint(nA)

        next_state, reward, done = env.step(curr_state, a)

        Q[curr_state, a] = Q[curr_state, a] + step_size * (
                env.reward(curr_state) + gamma * Q[next_state, np.argmax(Q[next_state])] - Q[curr_state, a])

        # reset environment if done or if horizon has been reached
        if done or t == horizon:
            end_reward = env.reward(next_state)
            # keep all state-action values the same for terminal states since there is no difference between actions
            # taken in terminal states
            Q[next_state] = Q[next_state] + step_size * (end_reward - Q[next_state])
            curr_state = env.reset(s_start=None)
            t = 0
            num_trajs += 1
            if num_trajs > traj_limit:
                break
        else:
            curr_state = next_state

    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)
    return V, Q, policy, num_trajs


if __name__ == '__main__':
    np.random.seed(2)
    from tests import BrownNiekum, Random
    from algorithms.value_iteration import value_iteration

    test = BrownNiekum()
    test.env.render()
    value_function_opt, policy = value_iteration(test.env)

    value_function_est, _, Q_policy = q_learning(test.wrapper, **{'n_samp': 50000, 'step_size': 0.1, 'epsilon': 0.1})


    # compare value functions
    print('Optimal policy: {}\nPolicy from Q-learning: {}'.format(policy, Q_policy))
    print('Optimal value function from Value Iteration: {}'.format(value_function_opt))
    print('Estimated value function from {}: {}'.format(q_learning.__name__, value_function_est))
