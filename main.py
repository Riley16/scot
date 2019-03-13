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

def main(args):
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
    else:
        test = BrownNiekum()

    print("{} grid environment:".format(test.__class__.__name__))
    test.env.render()

    if args.mc:
        test_mc(test.wrapper)
    else:
        SCOT(test.env, None, test.env.weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', choices=['basic', 'multiple', 'niekum'], default='niekum')
    parser.add_argument('-mc', action='store_true')
    args = parser.parse_args()

    start = time.time()
    main(args)
    end = time.time()
    print('Total run time: {}'.format(end - start))