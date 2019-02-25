import numpy as np
import argparse
from wrapper import Wrapper
from agent import Agent
from env import Grid
from actions import ACTIONS
from util_algo import *
from SCOT import SCOT

def main():
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

    # deterministic policy for testing
    # agent moves from right to left across grid to rightmost edge and then moves down to lower right corner
    # policy = np.zeros((args.height*args.width, 5))
    # policy[0:2, 3] = 1.0
    # policy[3:5, 3] = 1.0
    # policy[6:9, 3] = 1.0
    # policy[2, 4] = 1.0
    # policy[5, 4] = 1.0
    # print(policy)
    # policy = None

    np.random.seed(2)
    # def __init__(self, height:int, width:int, gamma:float, gray_sq:List[List[int]],
    #     gray_r:float, white_r:float, weights=None, num_feat:int=2, start_corner=True, start_dist=None):

    # env = Grid(args.height, args.width, args.gamma, gray_r=args.gray_r, white_r=args.white_r, start_corner=False)
    # agent = Agent(policy, args.height*args.width, len(ACTIONS))
    # env_wrapper = Wrapper(env, agent, log=True)
    # total_r, trajectories = env_wrapper.eval_episodes(args.n_episodes)
    # total_r, trajectories = env_wrapper.eval_episodes(args.n_episodes)

    # IS NOISE NEEDED FOR NON-SINGULAR MATRICES IN COMPUTATION OF BEC CONSTRAINTS?
    env_Niekum = Grid(2, 3, 0.9, gray_r=-2, white_r=-1, start_corner=False, gray_sq=[[1, 1]], noise=0.0)
    _, policy = value_iteration(env_Niekum)
    print(policy)
    policy = det2stoch_policy(policy, env_Niekum.nS, env_Niekum.nA)
    agent_Niekum = Agent(policy, env_Niekum.nS, env_Niekum.nA)
    env_wrapper_Niekum = Wrapper(env_Niekum, agent_Niekum, log=True)
    print("Grid environment:")
    env_Niekum.render()
    # total_r, trajectories = env_wrapper_Niekum.eval_episodes(args.n_episodes)
    #
    # print("Average reward with initial stochastic policy across {} episodes: {}".format(
    #     args.n_episodes, sum(total_r) / float(len(total_r))))

    # print(env_Niekum.weights.shape[0])
    # print(env.P)
    # print(agent.policy)
    # print(env.get_pol_trans(agent.policy))
    # print(policy)
    det_pol = stoch2det_policy(policy, env_Niekum.nS)
    # print(det_pol)
    stoch_pol = det2stoch_policy(det_pol, env_Niekum.nS, env_Niekum.nA)
    # print(stoch_pol)


    SCOT(env_Niekum, None, env_Niekum.weights)

if __name__ == "__main__":
    main()