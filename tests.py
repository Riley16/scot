import numpy as np
import argparse
from wrapper import Wrapper
from agent import Agent
from env import Grid
from actions import ACTIONS
from util_algo import *
from SCOT import SCOT

parser = argparse.ArgumentParser()
parser.add_argument('-width', type=int, default=3)
parser.add_argument('-height', type=int, default=3)
parser.add_argument('-gamma', type=float, default=1.0)
parser.add_argument('-gray_r', type=float, default=-10.0)
parser.add_argument('-white_r', type=float, default=-1.0)
parser.add_argument('-term_r', type=float, default=10.0)
parser.add_argument('-n_episodes', type=float, default=5)
args = parser.parse_args()

print(args)


# Vincent's square grid environment
# deterministic policy for testing
# agent moves from right to left across grid to rightmost edge and then moves down to lower right corner
policy_Vincent = np.zeros((args.height * args.width, 5))
policy_Vincent[0:2, 3] = 1.0
policy_Vincent[3:5, 3] = 1.0
policy_Vincent[6:9, 3] = 1.0
policy_Vincent[2, 4] = 1.0
policy_Vincent[5, 4] = 1.0
# print(policy_Vincent)
policy_Vincent = None
features_Vincent = [{
        'color': 'gray',
        'reward': 10.0,
        'squares': None
    }]

env = Grid(args.height, args.width, args.gamma, white_r=args.white_r, features_sq=features_Vincent, start_corner=False)
agent = Agent(policy_Vincent, args.height * args.width, len(ACTIONS))
env_wrapper = Wrapper(env, agent, log=True)
total_r, trajectories = env_wrapper.eval_episodes(args.n_episodes)
total_r, trajectories = env_wrapper.eval_episodes(args.n_episodes)


# Brown and Niekum toy environment (2019)
# our SCOT implementation currently handles cases of
#   gray_r    white_r
#   -10         -1
#   -2          -1
#   -1          -1
#   -1          -2
# does not handle cases of
#   0           0  (may not be necessary)
#   1           1  (may not be necessary)

# pass in a list of dicts for non-white square colors
features_Niekum = [{
        'color': 'gray',
        'reward': -10.0,
        'squares': [[1,1]]
    }]

#  env_Niekum = Grid(2, 3, 0.9, gray_r=-10, white_r=-1, start_corner=False, gray_sq=[[1, 1]], noise=0.0)
env_Niekum = Grid(2, 3, 0.9, white_r=-1, features_sq=features_Niekum, noise=0.0, start_corner=False)
_, policy_Niekum = value_iteration(env_Niekum)
print(policy_Niekum)
policy_Vincent = det2stoch_policy(policy_Niekum, env_Niekum.nS, env_Niekum.nA)
agent_Niekum = Agent(policy_Niekum, env_Niekum.nS, env_Niekum.nA)
env_wrapper_Niekum = Wrapper(env_Niekum, agent_Niekum, log=True)

