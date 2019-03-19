import numpy as np

# average lists
def avg_lists(lists):
    global_sum = 0
    lens = 0
    for l in lists:
        global_sum += sum(l)
        lens += len(l)
    print(global_sum / lens)

def var_lists(lists):
    variances = []
    for l in lists:
        variances.append(np.var(l))
    print(sum(variances) / len(lists))


"""
SARAH, THIS IS THE LIST OF TESTS THAT RILEY THINKS WOULD BE GOOD. DON'T FEEL LIKE THESE ARE OBVIOUSLY THE ONLY TESTS 
NEEDED, ETC., ETC.

we'll want to run tests on several grids. Here's the list of environments we want:
0. for our internal testing: Brown and Niekum (although maybe we could include this in the paper)
1. elongated corridor: 5-by-5 grid. Start anywhere. positive reward for reaching goal state in lower-right corner. 
                    zero reward for moving along upper row and rightmost column (the "corridor"), 
                    negative reward everywhere else (the inner 4-by-4 grid of the lower-left corner), 
                    might consider making leftmost column also part of the corridor
2. random grids with random numbers of features. probably want to vary size and num features separately. 
                    might want to try one-hot and distributed (not one-hot) features
                    The specific values don't matter too much, so long as they doesn't take too long
                    probably something like x-by-x for x from 1 to 12 with 8 random features,
                    and then like 8-by-8 or so (might be a little large) for 2 to like 15 random features

as for the tests themselves, as I see it, we're mainly comparing the sample complexity of SCOT with Maximum Likelihood 
IRL (MLIRL) to MLIRL with random trajectories from the teacher (as opposed to SCOT's optimally informative 
trajectories) along with various other methods of normal RL that we've implemented 
    1. PI with MC, first/every-visit, and TD learning as the policy evaluation methods. MAKE SURE PI IS WORKING WITH ALL OF THEM AS EXPECTED
    2. Q-learning

For the above environments and these methods, we want to evaluate performance, measured as policy similarity (proportion
of actions the same between policies across states) and more importantly, total value (expectation of the value function
over the start state distribution). These performance metrics should be measured against number of trajectories (each 
algorithm should have the same horizon of the number of states I think, although that might make MLIRL perform terribly, 
we can change that pretty easily, just have one horizon variable input to all of the algorithms) as well as number of 
(s, a, r, s') experience tuples. These are the main performance tests we're interested in. Have a single noise parameter
(and in general, make all of the shared parameters the same) for all of the environments, we might want to vary it, but
I think we're mostly interested in the deterministic case (although some of the algos will work way better with 
stochastic environments). We have a lot of methods and environments, we might want to just include these performance metrics
for the corridor env, and like one or two random environments with summary statistics for the others, like average number
of trajectories to convergence within some criterion, like 90% of the optimal total value or 0.90 policy similarity 
(some of the methods may not converge to this without a ton of samples). All of these performance tests need to be
averaged over ten randomized test runs (MAKE SURE THE DIFFERENT INTIALIZATIONS ARE INDEED GETTING DIFFERENT VALUES, 
MAKE SURE THE SEEDING IS LOCKED SO THAT THE TESTING IS REPEATABLE, MAYBE ITERATE THE RANDOM SEEDING OVER THE TEST RUNS).

In addition to these, we wanted to test the performance of our SCOT implementation by just measuring the runtime as well 
as mean and variance of number of trajectories and trajectory length (we need to make sure we communicate in the report 
that our horizon differed across the environments as the number of states in the environment, or some multiple of it 
depending on noise and other factors...). I'm probably forgetting some of the other performance metrics of our SCOT
implementation, but you know what we were planning for this. We'll probably just want tables for these values. We'll 
probably want to test these statistics for like 5 runs on 10 different random initializations, although that might take 
longer.

Finally, as Vincent was saying, we could test SCOT and MLIRL with suboptimal policies (maybe on larger environments) 
from PI as a simple test of applying SCOT and MLIRL to problems with more complex environments where optimal teachers 
can only be approximated.

"""




if __name__ == "__main__":
    lists = [
        [34, 16, 14, 19, 76, 21, 18, 18, 79, 29, 25, 28, 24, 25, 73, 79, 21, 16, 75, 15]
        ]
    avg_lists(lists)
    var_lists(lists)
