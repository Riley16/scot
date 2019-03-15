import numpy as np
import argparse
from tests import *
from algorithms.scot import scot
from algorithms.value_iteration import value_iteration
from algorithms.max_likelihood_irl import max_likelihood_irl

def test_mc(wrapper):
    ''' Evaluate Monte Carlo algorithm. '''

    # generate optimal policy / value function from value iteration
    value_function_vi, policy = value_iteration(wrapper.env)
    
    # evaluate optimal policy with monte carlo
    value_function_mc = monte_carlo(wrapper, n=-1, eps=1e-5)

    # compare value functions
    print('Optimal policy: {}'.format(policy))
    print('Value function from VI: {}'.format(value_function_vi))
    print('Value function from MC: {}'.format(value_function_mc))

    return value_function_vi, value_function_mc

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
    test = get_env() # default BrownNiekum()
    test.env.render()

    np.random.seed(2)
    trajs = scot(test.env, test.env.weights, verbose=True)

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
    
