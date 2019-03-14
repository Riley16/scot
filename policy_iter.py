import numpy as np
import env
from wrapper import Wrapper
from agent import Agent
from env import Grid
from actions import ACTIONS
from util_algo import *
from SCOT import SCOT


"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	value_function = np.zeros(nS)
	############################
	# YOUR IMPLEMENTATION HERE #
	i = 0
	delta = 0
	while i == 0 or delta > tol: # is this the correct way to do l1 norm?
		delta = 0
		old_value_function = np.copy(value_function)
		for s in range(nS):
			new_value = 0
			for prob_of_trans, next_s, reward, terminal in P[s][policy[s]]:
				#print(prob_of_trans, next_s, reward, terminal)
				new_value += prob_of_trans * (reward + gamma * old_value_function[next_s])
			value_function[s] = new_value
			delta = max(delta, abs(old_value_function[s] - new_value)) #compute new delta
		i += 1
	############################
	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
	new_policy = np.zeros(nS, dtype='int')
	############################
	# YOUR IMPLEMENTATION HERE #
	for s in range(nS):
		# calculate Q value for taking all actions to see if currently policy is correct
		q_values = []
		for a in range(nA):
			action_value = 0
			for prob_of_trans in P[s][a]:
				# get next state
				grid_position = env.state_to_grid(s)
				grid_move = env.actions_to_grid[a]
				new_grid_position = grid_position + grid_move
                next_s = env.grid_to_state(new_grid_position)
				# get reward
				reward = env.reward(next_s)
				terminal = env.is_terminal(next_s)
				action_value += (prob_of_trans * (reward + gamma * value_from_policy[next_s]))
			q_values.append(action_value)
		new_policy[s] = q_values.index(max(q_values)) # find the action corresponding to largest Q

	############################
	return new_policy


def policy_iteration(policy_evaluation_function, P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)

	############################
	# YOUR IMPLEMENTATION HERE #
	i = 0
	# initialize initial policy randomly for all states
	old_policy = np.copy(policy)
	while i == 0 or np.array_equal(old_policy, policy) == False:
		"""
		VINCENT: I want to be able to pass your MC function in as policy_evaluation_function
		Your function should take in P which is the transition function, policy, and everything
		else below OR at least generate those things within the function (in which case I can 
		just not pass them in here. Let me know
		"""
		value_function = policy_evaluation_function(P, nS, nA, policy, gamma, tol)
		old_policy = np.copy(policy)
		policy = policy_improvement(P, nS, nA, value_function, policy, gamma)
		i += 1
	############################
	return value_function, policy

if __name__ == "__main__":
	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)

