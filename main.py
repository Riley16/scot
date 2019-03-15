import numpy as np
import argparse
from wrapper import Wrapper
from agent import Agent
from env import Grid
from actions import ACTIONS
from util_algo import *
from SCOT import SCOT
from tests import *
import time

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', choices=['basic', 'multiple', 'niekum', 'paper_test'], default='niekum')
    args = parser.parse_args()
    return args.env



def main():
    # env = parse_arguments()
    #
    #
    # total_sum = []
    # num_trajs = []
    # all_lens = []
    # times = []
    # numTests = 1
    # for i in range(numTests):
    #     if env == 'basic':
    #         test = BasicGrid()
    #     elif env == 'multiple':
    #         test = MultipleFeatures()
    #     elif env == "niekum":
    #         test = BrownNiekum()
    #     elif env == "cooridor":
    #         test = Cooridor()
    #     elif env == "loop":
    #         test = Loop()
    #     elif env == "paper_test":
    #         test = FromPaper()
    #     else:
    #         test = BrownNiekum()
        # print("{} grid environment:".format(test.__class__.__name__))

        # test = BrownNiekum()
        # test.env.render()

    #     print("\n ITER", i)
    #     t0 = time.time()
    #     D, lens = SCOT(test.env, None, test.env.weights)
    #     num_trajs.append(len(D))
    #     total_sum.append(sum(lens))
    #     all_lens += lens
    #     print(len(D), lens)
    #     trial_time = time.time() - t0
    #     times.append(trial_time)
    #
    # avg_num_trajs = np.mean(num_trajs)
    # avg_traj_lens = sum(total_sum) / sum(num_trajs)
    # var_num_trajs = np.var(num_trajs)
    # var_traj_lens = np.var(all_lens)
    # avg_time = np.mean(times)
    # var_time = np.var(times)
    #
    # print("avg_num_trajs, avg_traj_lens, var_num_trajs, var_traj_lens \n",
    #       avg_num_trajs, avg_traj_lens, var_num_trajs, var_traj_lens)
    # print("avg_time, var_time\n", avg_time, var_time)
    #
    # print("{} grid environment:".format(test.__class__.__name__))

    test = BrownNiekum()
    test.env.render()

    np.random.seed(2)
    trajs = SCOT(test.env, test.env.weights, verbose=True)
    # student's inferred reward function from the trajectories from SCOT
    r_weights = maxLikelihoodIRL(trajs, test.env, step_size=0.2, eps=1.0e-03, max_steps=1000, verbose=True)
    # r_weights = maxLikelihoodIRL(trajs, test.env, step_size=0.001, eps=1.0e-02, max_steps=1000, verbose=True)

    # print("teacher reward weights")
    # print(test.env.weights)
    # print('learned reward weights')
    # print(r_weights)

    values_MLIRL, policy_MLIRL = value_iteration(mdp=test.env, r_weights=r_weights) # student's policy and value function under student's reward funct
    values_MLIRL = value_iteration(mdp=test.env, policy=policy_MLIRL) # value of student's policy under teacher's reward funct (true)
    values_opt, policy_opt = value_iteration(mdp=test.env) # optimal value and policy under teacher's reward funct (true)
    # print(policy_MLIRL)
    # print(policy_opt)
    policy_similarity = np.sum(policy_MLIRL == policy_opt)/policy_MLIRL.shape[0]

    # policy similarity
    print("Policy similarity:")
    print(policy_similarity)
    # print(values_MLIRL)
    # print(values_opt)

    # value gain
    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt)
    print("optimal expected value")
    print(total_value_opt)

    total_value_MLIRL = np.dot(test.env.start_dist, values_MLIRL)
    print("MLIRL expected value")
    print(total_value_MLIRL)

    value_gain_MLIRL = total_value_MLIRL/total_value_opt
    print("Value gain of MLIRL")
    print(value_gain_MLIRL)


if __name__ == "__main__":
    main()
    