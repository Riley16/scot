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
    trajs = SCOT(test.env, None, test.env.weights)
    r_weights = maxLikelihoodIRL(trajs, test.env, step_size=0.001, eps=1.0e-03)
    print(test.env.weights)
    print(r_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', choices=['basic', 'multiple', 'niekum', 'paper_test'], default='niekum')
    args = parser.parse_args()
    print()
    t0 = time.time()
    main()
    print(time.time() - t0)