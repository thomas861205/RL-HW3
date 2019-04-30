"""
Description:
    You are going to implement Dyna-Q, a integration of model-based and model-free methods. 
    Please follow the instructions to complete the assignment.
"""
import numpy as np 
from copy import deepcopy

def choose_action(state, q_value, maze, epislon):
    """
    Description:
        choose the action using epislon-greedy policy
    """
    if np.random.random() < epislon:
        return np.random.choice(maze.actions)
    else:
        values = q_value[state[0], state[1], :]
        return np.random.choice([action for action, value in enumerate(values) if value == np.max(values)])

def dyna_q(args, q_value, model, maze):
    """
    Description:
        Dyna-Q algorithm is here :)
    Inputs:
        args:    algorithm parameters
        q_value: Q table to maintain.
        model:   The internal model learned by Dyna-Q 
        maze:    Maze environment
    Return:
        steps:   Total steps taken in an episode.
    TODO:
        Complete the algorithm.
    """
    state = maze.START_STATE # type: list
    steps = 0
   
    # raise NotImplementedError("Dyna-Q NOT IMPLEMENTED")
    while state != [0, 8]:
        steps += 1
        action = choose_action(state, q_value, maze, args.epislon)
        next_state, reward = maze.step(tuple(state), action)
        # print("State {} -> Action {} -> Next State {}".format(state, action, next_state))
        target = reward + args.gamma * max(q_value[next_state[0], next_state[1], :])
        q_value[state[0], state[1], action] = q_value[state[0], state[1], action] + args.alpha * (target - q_value[state[0], state[1], action])
        
        model.store(state, action, next_state, reward)
        state = next_state

        for n in range(args.planning_steps):
            (stateP, actionP, next_stateP, rewardP) = model.sample()
            targetP = rewardP + args.gamma * max(q_value[next_stateP[0], next_stateP[1], :])
            q_value[stateP[0], stateP[1], actionP] = q_value[stateP[0], stateP[1], actionP] + args.alpha * (targetP - q_value[stateP[0], stateP[1], actionP])


    return steps

class InternalModel(object):
    """
    Description:
        We'll create a tabular model for our simulated experience. Please complete the following code.
    """
    def __init__(self):
        # self.model = dict()
        self.model = []
        self.rand = np.random
    
    def store(self, state, action, next_state, reward):
        """
        TODO:
            Store the previous experience into the model.
        Return:
            NULL
        """        
        exp = (state, action, next_state, reward)
        self.model.append(exp)

    def sample(self):
        """
        TODO:
            Randomly sample previous experience from internal model.
        Return:
            state, action, next_state, reward
        """
        import random
        (stateP, actionP, next_stateP, rewardP) = random.sample(self.model, 1)[0]
        return (stateP, actionP, next_stateP, rewardP)
