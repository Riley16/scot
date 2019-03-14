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

    if args.env == 'basic':
        test = BasicGrid()
    elif args.env == 'multiple':
        test = MultipleFeatures()
    elif args.env == "niekum":
        test = BrownNiekum()
    elif args.env == "cooridor":
        test = Cooridor()
    elif args.env == "loop":
        test = Loop()
    elif args.env == "paper_test":
        test = MultipleFeatures_Test()
    else:
        test = BrownNiekum()
    return test


def main():
    #parse_arguments()

    #print("{} grid environment:".format(test.__class__.__name__))

    total_sum = []
    num_trajs = []
    all_lens = []
    times = []
    numTests = 20
    for i in range(numTests):
        test = BrownNiekum()
        test.env.render()

        print("\n ITER", i)
        t0 = time.time()
        D, lens = SCOT(test.env, None, test.env.weights)
        num_trajs.append(len(D))
        total_sum.append(sum(lens))
        all_lens += lens
        print(len(D), lens)
        trial_time = time.time() - t0
        times.append(trial_time)

    avg_num_trajs = np.mean(num_trajs)
    avg_traj_lens = sum(total_sum) / sum(num_trajs)
    var_num_trajs = np.var(num_trajs)
    var_traj_lens = np.var(all_lens)
    avg_time = np.mean(times)
    var_time = np.var(times)

    print("avg_num_trajs, avg_traj_lens, var_num_trajs, var_traj_lens \n",
          avg_num_trajs, avg_traj_lens, var_num_trajs, var_traj_lens)
    print("avg_time, var_time\n", avg_time, var_time)


if __name__ == "__main__":
    main()
