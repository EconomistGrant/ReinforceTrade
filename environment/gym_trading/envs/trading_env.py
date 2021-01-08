import random
import gym
import numpy as np

MAX = 10000
MIN = -10000
class TradingEnv(gym.Env):
    """
    A trading environment set up using OpenAI Gym
    Data are stored in numpy.array format
    """
    def __init__(self, arr: np.array, back_looking = 1: int):
        super(TradingEnv,self).__init__()

        #TODO: standardize input
        self.arr = arr
        self.back_looking = back_looking
        self.current_step = 0

        assert len(arr.shape) == 2, 'check input data is two-dimensional'
        assert arr.shape[0] > 0, 'check data length'
        num_obervation = arr.shape[0]
        num_features = arr.shape[1]

        # first number: 0 = buy, 1 = sell, 2 = hold; second number: percent of holding as trade size
        #TODO: is there any better way of defining action space?
        self.action_space = gym.spaces.Box(low = np.array([0,0]), high = np.array([3,1]), dtype = np.float16) 

        #all features from input + current holding
        #TODO: including current NAV: even more interesting?
        self.observation_space =gym.spaces.Box(low = MIN, high = MAX, shape = (back_looking,num_features+1))

    def step(self,action):
        ...

    def reset(self):
        ...
    def render(self, mode = 'human', close = False)
        ...