import numpy as np
import argparse
from tests import *
from algorithms.scot import scot
from algorithms.value_iteration import value_iteration
from algorithms.max_likelihood_irl import max_likelihood_irl
from algorithms.policy_iteration import policy_iteration
from algorithms.policy_evaluation import temporal_difference, every_visit_monte_carlo, first_visit_monte_carlo

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
    elif env == "random":
        test = Random()
    else:
        test = BrownNiekum()

    return test


def test_PI(test, policy_opt, values_opt, horizon):
    est_values_PI, policy_PI = policy_iteration(
        test.env, test.agent, every_visit_monte_carlo, kwargs={'n_eps': 50, 'eps_len': 10})
    values_PI, _ = value_iteration(mdp=test.env,
                                   policy=policy_PI)  # value of student's PI policy under teacher's reward funct (true)

    policy_similarity = np.sum(policy_PI == policy_opt) / policy_PI.shape[0]

    #print("Policy similarity: {}".format(policy_similarity))

    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt)
    #print("Optimal expected value: {}".format(total_value_opt))

    total_value_PI = np.dot(test.env.start_dist, values_PI)
    #print("True PI expected value: {}".format(total_value_PI))

    value_gain_PI = total_value_PI / total_value_opt
    #print("Value gain of true PI: {}".format(value_gain_PI))

    total_value_est_PI = np.dot(test.env.start_dist, est_values_PI)
    #print("Estimated PI expected value: {}".format(total_value_est_PI))

    value_gain_est_PI = total_value_est_PI / total_value_opt
    #print("Value gain of Est PI: {}".format(value_gain_est_PI))

    return policy_similarity, value_gain_PI, value_gain_est_PI


def test_scot(test, policy_opt, values_opt, seed, horizon):
    trajs = scot(test.env, test.env.weights, H=horizon, seed=seed, verbose=False)
    print(len(trajs))
    exit(0)
    # student's inferred reward function from the trajectories from SCOT
    r_weights = max_likelihood_irl(trajs, test.env, step_size=0.2, eps=1.0e-03, max_steps=1000, verbose=False)

    values_MLIRL, policy_MLIRL = value_iteration(mdp=test.env,
                                                 r_weights=r_weights)  # student's policy and value function under student's reward funct
    values_MLIRL, _ = value_iteration(mdp=test.env,
                                      policy=policy_MLIRL)  # value of student's policy under teacher's reward funct (true)

    policy_similarity = np.sum(policy_MLIRL == policy_opt) / policy_MLIRL.shape[0]

    #print("Policy similarity: {}".format(policy_similarity))

    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt)
    #print("Optimal expected value: {}".format(total_value_opt))

    total_value_MLIRL = np.dot(test.env.start_dist, values_MLIRL)
    #print("Max Likelihood IRL expected value: {}".format(total_value_MLIRL))

    value_gain_MLIRL = total_value_MLIRL / total_value_opt
    #print("Value gain of Max Likelihood IRL: {}".format(value_gain_MLIRL))
    return policy_similarity, value_gain_MLIRL


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

    values_opt, policy_opt = value_iteration(mdp=test.env) # optimal value and policy under teacher's reward funct (true)

    horizon = 100
    num_tests = 10
    MLIRL_policy_similarity_list = []
    value_gain_MLIRL_list = []
    PI_policy_similarity_list = []
    value_gain_PI_list = []
    value_gain_est_PI_list = []
    for _ in range(num_tests):
        MLIRL_policy_similarity, value_gain_MLIRL = test_scot(test, policy_opt, values_opt, seed, horizon)
        PI_policy_similarity, value_gain_PI, value_gain_est_PI = test_PI(test, policy_opt, values_opt, horizon)

        MLIRL_policy_similarity_list.append(MLIRL_policy_similarity)
        value_gain_MLIRL_list.append(value_gain_MLIRL)
        PI_policy_similarity_list.append(PI_policy_similarity)
        value_gain_PI_list.append(value_gain_PI)
        value_gain_est_PI_list.append(value_gain_est_PI)

    print("MLIRL_policy_similarity", MLIRL_policy_similarity_list)
    print("value_gain_MLIRL", value_gain_MLIRL_list)
    print("PI_policy_similarity", PI_policy_similarity_list)
    print("value_gain_PI", value_gain_PI_list)
    print("value_gain_est_PI", value_gain_est_PI_list)




if __name__ == "__main__":
    main()
    
