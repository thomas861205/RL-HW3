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
    state = maze.START_STATE
    steps = 0
   
    raise NotImplementedError("Dyna-Q NOT IMPLEMENTED")
    
    return steps

class InternalModel(object):
    """
    Description:
        We'll create a tabular model for our simulated experience. Please complete the following code.
    """
    def __init__(self):
        self.model = dict()
        self.rand = np.random
    
    def store(self, state, action, next_state, reward):
        """
        TODO:
            Store the previous experience into the model.
        Return:
            NULL
        """
        
        raise NotImplementedError('InternalModel NOT IMPLEMENTED')

    def sample(self):
        """
        TODO:
            Randomly sample previous experience from internal model.
        Return:
            state, action, next_state, reward
        """
        raise NotImplementedError('InternalModel NOT IMPLEMENTED')
