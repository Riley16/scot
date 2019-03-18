import numpy as np
import argparse
from tests import *
from algorithms.scot import scot
from algorithms.value_iteration import value_iteration
from algorithms.max_likelihood_irl import max_likelihood_irl

def get_env():
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', default='niekum') # choices=['basic', 'multiple', 'cooridor', 'paper_test', 'niekum']
    args = parser.parse_args()
    env = args.env

    if env == 'basic':
        test = BasicGrid()
    elif env == 'multiple':
        test = MultipleFeatures()
    elif env == "niekum":
        test = BrownNiekum()
    elif env == "cooridor":
        test = Cooridor()
    elif env == "loop":
        test = Loop()
    elif env == "paper_test":
        test = FromPaper()
    else:
        test = BrownNiekum()

    return test


def main():
    seed = 2
    np.random.seed(seed)
    test = get_env()  # default BrownNiekum()
    test.env.render()
    # for i in range(50, 100):
    #     np.random.seed(i)
    #     test = get_env()  # default BrownNiekum()
    #     print("i")
    #     print(i)
    #     trajs = scot(test.env, test.env.weights, seed=i+1, verbose=False)

    trajs = scot(test.env, test.env.weights, seed=seed, verbose=False)
    # student's inferred reward function from the trajectories from SCOT
    r_weights = max_likelihood_irl(trajs, test.env, step_size=0.2, eps=1.0e-03, max_steps=1000, verbose=True)

    values_MLIRL, policy_MLIRL = value_iteration(mdp=test.env, r_weights=r_weights) # student's policy and value function under student's reward funct
    values_MLIRL, _ = value_iteration(mdp=test.env, policy=policy_MLIRL) # value of student's policy under teacher's reward funct (true)
    values_opt, policy_opt = value_iteration(mdp=test.env) # optimal value and policy under teacher's reward funct (true)
    policy_similarity = np.sum(policy_MLIRL == policy_opt)/policy_MLIRL.shape[0]

    print("Policy similarity: {}".format(policy_similarity))

    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt)
    print("Optimal expected value: {}".format(total_value_opt))

    total_value_MLIRL = np.dot(test.env.start_dist, values_MLIRL)
    print("Max Likelihood IRL expected value: {}".format(total_value_MLIRL))

    value_gain_MLIRL = total_value_MLIRL/total_value_opt
    print("Value gain of Max Likelihood IRL: {}".format(value_gain_MLIRL))


if __name__ == "__main__":
    main()
    
