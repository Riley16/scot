import numpy as np
import argparse
from tests import *
from algorithms.scot import scot
from algorithms.value_iteration import value_iteration
from algorithms.max_likelihood_irl import max_likelihood_irl
from algorithms.policy_iteration import policy_iteration
from algorithms.policy_evaluation import temporal_difference, every_visit_monte_carlo, first_visit_monte_carlo
from algorithms.q_learning import q_learning
from algorithms.baseline import baseline
from random import sample

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

def test_QLearning(test, policy_opt, values_opt, horizon, num_samples):
    value_function_est, _, Q_policy = q_learning(test.wrapper, **{'n_samp': num_samples, 'step_size': 0.1, 'epsilon': 0.1, 'horizon':horizon})

    values_QL, _ = value_iteration(mdp=test.env,
                                   policy=Q_policy)  # value of student's PI policy under teacher's reward funct (true)

    policy_similarity = np.sum(Q_policy == policy_opt) / Q_policy.shape[0]
    print("Policy similarity for Q Learning: {}".format(policy_similarity))


    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt)
    #print("Optimal expected value: {}".format(total_value_opt))

    total_value_QL = np.dot(test.env.start_dist, values_QL)
    print("True QL expected value: {}".format(total_value_QL))

    value_gain_QL = total_value_QL / total_value_opt
    print("Value gain of true PI: {}".format(value_gain_QL))

    total_value_est_QL = np.dot(test.env.start_dist, value_function_est)
    print("Estimated PI expected value: {}".format(total_value_est_QL))

    value_gain_est_QL = total_value_est_QL / total_value_opt
    print("Value gain of Est QL: {}".format(value_gain_est_QL))

    return policy_similarity, total_value_QL, total_value_est_QL, value_gain_QL, value_gain_est_QL

def test_PI(test, policy_opt, values_opt, horizon, num_samples):
    print("testPI")
    #est_values_PI, policy_PI = policy_iteration(
        #test.env, test.agent, every_visit_monte_carlo, kwargs={'n_eps': 50, 'eps_len': horizon})

    #est_values_PI, policy_PI = policy_iteration(
     #   test.env, test.agent, temporal_difference, kwargs={'n_samp':1000, 'step_size': 0.1, 'horizon': horizon})

    est_values_PI, policy_PI = policy_iteration(
        test.env, test.agent, first_visit_monte_carlo, kwargs={'n_eps': int(num_samples / horizon), 'eps_len': horizon})
    print('here')


    values_PI, _ = value_iteration(mdp=test.env,
                                   policy=policy_PI)  # value of student's PI policy under teacher's reward funct (true)

    policy_similarity = np.sum(policy_PI == policy_opt) / policy_PI.shape[0]
    print("Policy similarity for PI: {}".format(policy_similarity))

    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt)
    #print("Optimal expected value: {}".format(total_value_opt))

    total_value_PI = np.dot(test.env.start_dist, values_PI)
    print("True PI expected value: {}".format(total_value_PI))

    value_gain_PI = total_value_PI / total_value_opt
    print("Value gain of true PI: {}".format(value_gain_PI))

    total_value_est_PI = np.dot(test.env.start_dist, est_values_PI)
    print("Estimated PI expected value: {}".format(total_value_est_PI))

    value_gain_est_PI = total_value_est_PI / total_value_opt
    print("Value gain of Est PI: {}".format(value_gain_est_PI))

    return policy_similarity, total_value_PI, total_value_est_PI, value_gain_PI, value_gain_est_PI

def test_baseline(test, policy_opt, values_opt, seed, horizon, num_samples):
    samples = baseline(test.env, test.agent, num_samples, num_samples * 2, horizon)
    lens = []
    for t in samples:
        lens.append(len(t))
    # student's inferred reward function from the trajectories from SCOT
    r_weights = max_likelihood_irl(samples, test.env, step_size=0.2, eps=1.0e-03, max_steps=1000, verbose=False)

    values_MLIRL, policy_MLIRL = value_iteration(mdp=test.env,
                                                 r_weights=r_weights)  # student's policy and value function under student's reward funct
    values_MLIRL, _ = value_iteration(mdp=test.env,
                                      policy=policy_MLIRL)  # value of student's policy under teacher's reward funct (true)

    policy_similarity = np.sum(policy_MLIRL == policy_opt) / policy_MLIRL.shape[0]

    print("Policy similarity for Baseline: {}".format(policy_similarity))

    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt)
    # print("Optimal expected value: {}".format(total_value_opt))

    total_value_MLIRL = np.dot(test.env.start_dist, values_MLIRL)
    print("Max Likelihood IRL expected value: {}".format(total_value_MLIRL))

    value_gain_MLIRL = total_value_MLIRL / total_value_opt
    print("Value gain of Max Likelihood IRL: {}".format(value_gain_MLIRL))
    return policy_similarity, total_value_MLIRL, value_gain_MLIRL


def test_scot(test, policy_opt, values_opt, seed, horizon, num_samples):
    trajs = scot(test.env, test.env.weights, H=horizon, seed=seed, verbose=False)
    lens = []
    for t in trajs:
        lens.append(len(t))
    print("num_samples", num_samples)
    if len(trajs) > num_samples:
        trajs = sample(trajs, num_samples)
    # student's inferred reward function from the trajectories from SCOT
    r_weights = max_likelihood_irl(trajs, test.env, step_size=0.2, eps=1.0e-03, max_steps=1000, verbose=False)

    values_MLIRL, policy_MLIRL = value_iteration(mdp=test.env,
                                                 r_weights=r_weights)  # student's policy and value function under student's reward funct
    values_MLIRL, _ = value_iteration(mdp=test.env,
                                      policy=policy_MLIRL)  # value of student's policy under teacher's reward funct (true)

    policy_similarity = np.sum(policy_MLIRL == policy_opt) / policy_MLIRL.shape[0]

    print("Policy similarity for SCOT: {}".format(policy_similarity))

    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt)
    #print("Optimal expected value: {}".format(total_value_opt))

    total_value_MLIRL = np.dot(test.env.start_dist, values_MLIRL)
    print("Max Likelihood IRL expected value: {}".format(total_value_MLIRL))

    value_gain_MLIRL = total_value_MLIRL / total_value_opt
    print("Value gain of Max Likelihood IRL: {}".format(value_gain_MLIRL))
    print(len(trajs), sum(lens))
    return policy_similarity, total_value_MLIRL, value_gain_MLIRL


def main():

    # for i in range(50, 100):
    #     np.random.seed(i)
    #     test = get_env()  # default BrownNiekum()
    #     print("i")
    #     print(i)
    #     trajs = scot(test.env, test.env.weights, seed=i+1, verbose=False)


    horizon = 20
    num_samples = 40
    #print(test_scot(test, policy_opt, values_opt, seed, horizon))
    #print(test_PI(test, policy_opt, values_opt, horizon))

    num_tests = 10
    MLIRL_policy_similarity_list = []
    total_value_MLIRL_list = []
    value_gain_MLIRL_list = []

    PI_policy_similarity_list = []
    total_value_PI_list = []
    total_value_est_PI_list = []
    value_gain_PI_list = []
    value_gain_est_PI_list = []

    QL_policy_similarity_list = []
    total_value_QL_list = []
    total_value_est_QL_list = []
    value_gain_QL_list = []
    value_gain_est_QL_list = []

    baseline_policy_similarity_list = []
    total_value_baseline_list = []
    value_gain_baseline_list = []

    for i in range(num_tests):
        np.random.seed(i)
        test = get_env()  # default BrownNiekum()
        test.env.render()
        values_opt, policy_opt = value_iteration(
            mdp=test.env)  # optimal value and policy under teacher's reward funct (true)

        print(i)
        MLIRL_policy_similarity, total_value_MLIRL, value_gain_MLIRL = test_scot(test, policy_opt, values_opt, i, horizon, int(num_samples/horizon))
        #PI_policy_similarity, total_value_PI, total_value_est_PI, value_gain_PI, value_gain_est_PI = test_PI(test, policy_opt, values_opt, horizon, int(num_samples/horizon))
        #QL_policy_similarity, total_value_QL, total_value_est_QL, value_gain_QL, value_gain_est_QL = test_QLearning(test, policy_opt, values_opt, horizon, num_samples)
        #baseline_policy_similarity, total_value_baseline, value_gain_baseline  = test_baseline(test, policy_opt, values_opt, i, horizon, int(num_samples/horizon))

        MLIRL_policy_similarity_list.append(MLIRL_policy_similarity)
        total_value_MLIRL_list.append(total_value_MLIRL)
        value_gain_MLIRL_list.append(value_gain_MLIRL)

        #PI_policy_similarity_list.append(PI_policy_similarity)
        #total_value_PI_list.append(total_value_PI)
        #total_value_est_PI_list.append(total_value_est_PI)
        #value_gain_PI_list.append(value_gain_PI)
        #value_gain_est_PI_list.append(value_gain_est_PI)
        """ 
        QL_policy_similarity_list.append(QL_policy_similarity)
        total_value_QL_list.append(total_value_QL)
        total_value_est_QL_list.append(total_value_est_QL)
        value_gain_QL_list.append(value_gain_QL)
        value_gain_est_QL_list.append(value_gain_est_QL)
        

        baseline_policy_similarity_list.append(baseline_policy_similarity)
        total_value_baseline_list.append(total_value_baseline)
        value_gain_baseline_list.append(value_gain_baseline)    
        """
    print("MLIRL_policy_similarity", np.mean(MLIRL_policy_similarity_list), np.var(MLIRL_policy_similarity_list))
    print("total_value_MLIRL", np.mean(total_value_MLIRL_list), np.var(total_value_MLIRL_list))
    print("value_gain_MLIRL", np.mean(value_gain_MLIRL_list), np.var(value_gain_MLIRL_list))

    #print("PI_policy_similarity", np.mean(PI_policy_similarity_list), np.var(PI_policy_similarity_list))
    #print("total_value_PI", np.mean(total_value_PI_list), np.var(total_value_PI_list))
    #print("total_value_est_PI", np.mean(total_value_est_PI_list), np.var(total_value_est_PI_list))
    #print("value_gain_PI", np.mean(value_gain_PI_list), np.var(value_gain_PI_list))
    #print("value_gain_est_PI", np.mean(value_gain_est_PI_list), np.var(value_gain_est_PI_list))

    print("QL_policy_similarity", np.mean(QL_policy_similarity_list), np.var(QL_policy_similarity_list))
    print("total_value_QL", np.mean(total_value_QL_list), np.var(total_value_QL_list))
    print("total_value_est_QL", np.mean(total_value_est_QL_list), np.var(total_value_est_QL_list))
    print("value_gain_QL", np.mean(value_gain_QL_list), np.var(value_gain_QL_list))
    print("value_gain_est_QL", np.mean(value_gain_est_QL_list), np.var(value_gain_est_QL_list))

    print("baseline_policy_similarity", np.mean(baseline_policy_similarity_list), np.var(baseline_policy_similarity_list))
    print("total_value_baseline", np.mean(total_value_baseline_list), np.var(total_value_baseline_list))
    print("value_gain_baseline", np.mean(value_gain_baseline_list), np.var(value_gain_baseline_list))


if __name__ == "__main__":
    main()
    
