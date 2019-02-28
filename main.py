import numpy as np
import argparse
from wrapper import Wrapper
from agent import Agent
from env import Grid
from actions import ACTIONS
from util_algo import *
from SCOT import SCOT
from tests import *

def main():
    np.random.seed(2)

    env = env_Niekum
    policy = policy_Niekum
    wrapper = env_wrapper_Niekum

    print("Grid environment:")
    env.render()

    SCOT(env_Niekum, None, env_Niekum.weights)

if __name__ == "__main__":
    main()