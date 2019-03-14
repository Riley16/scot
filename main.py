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
    # student's inferred reward function from the trajectories from SCOT
    r_weights = maxLikelihoodIRL(trajs, test.env, step_size=0.1, eps=1.0e-03, max_steps=1000, verbose=True) 
    print(test.env.weights)
    print(r_weights)
    values_MLIRL, policy_MLIRL = value_iteration(mdp=test.env, r_weights=r_weights) # student's policy and value function under student's reward funct
    values_MLIRL = value_iteration(mdp=test.env, policy=policy_MLIRL) # value of student's policy under teacher's reward funct (true)
    values_opt, policy_opt = value_iteration(mdp=test.env) # optimal value and policy under teacher's reward funct (true)
    print(policy_MLIRL)
    print(policy_opt)
    policy_similarity = np.sum(policy_MLIRL == policy_opt)/policy_MLIRL.shape[0]

    # policy similarity
    print("Policy similarity:")
    print(policy_similarity)
    print(values_MLIRL)
    print(values_opt)

    # value gain
    # FOR NOW, THE START STATE DISTRIBUTION START_DIST DOES NOT HAVE AN ELEMENT FOR THE STATE nS
    total_value_opt = np.dot(test.env.start_dist, values_opt[:-1])
    print("optimal expected value")
    print(total_value_opt)

    total_value_MLIRL = np.dot(test.env.start_dist, values_MLIRL[:-1])
    print("MLIRL expected value")
    print(total_value_MLIRL)

    value_gain_MLIRL = total_value_MLIRL/total_value_opt
    print("Value gain of MLIRL")
    print(value_gain_MLIRL)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', choices=['basic', 'multiple', 'niekum', 'paper_test'], default='niekum')
    args = parser.parse_args()
    print()
    t0 = time.time()
    main()
    print(time.time() - t0)
