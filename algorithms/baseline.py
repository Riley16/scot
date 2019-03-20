import numpy as np
from wrapper import Wrapper
from random import sample

def baseline(env, agent, num_samples, n_episodes, horizon):
    wrapper = Wrapper(env, agent, log=True)
    total_r, trajectories = wrapper.eval_episodes(n_episodes, horizon=horizon)

    #trajectories = np.array(trajectories, dtype=int)
    #indices = np.random.choice(len(trajectories), num_samples)
    #samples = trajectories[indices]
    #return list(samples)
    samples = (sample(trajectories,num_samples))
    return samples

if __name__ == '__main__':
    samples = baseline()
    print(samples)
    print(len(samples), type(samples))