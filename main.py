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

def main():
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

    print("{} grid environment:".format(test.__class__.__name__))
    test.env.render()

    np.random.seed(2)
    trajs = SCOT(test.env, None, test.env.weights, verbose=True)
    r_weights = maxLikelihoodIRL(trajs, test.env, step_size=0.1, eps=1.0e-03, max_steps=1000, verbose=True)
    print(test.env.weights)
    print(r_weights)
    values_MLIRL, policy_MLIRL = value_iteration(mdp=test.env, r_weights=r_weights)
    values_MLIRL = value_iteration(mdp=test.env, policy=policy_MLIRL)
    values_opt, policy_opt = value_iteration(mdp=test.env)
    print(policy_MLIRL)
    print(policy_opt)
    policy_similarity = np.sum(policy_MLIRL == policy_opt)/policy_MLIRL.shape[0]

    # policy similarity
    print(policy_similarity)
    print(values_MLIRL)
    print(values_opt)

    # value gain
    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    value_gain_opt = np.dot(test.env.start_dist, values_opt[:-1])
    print(value_gain_opt)
    value_gain_MLIRL = np.dot(test.env.start_dist, values_MLIRL[:-1])
    print(value_gain_MLIRL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', choices=['basic', 'multiple', 'niekum', 'paper_test'], default='niekum')
    args = parser.parse_args()
    print()
    t0 = time.time()
    main()
    print(time.time() - t0)