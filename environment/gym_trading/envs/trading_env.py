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
    def __init__(self, data: np.array, close_col: int = 1, back_looking :int = 1):
        super(TradingEnv,self).__init__()

        #TODO: standardize input
        
        self.data = data
        self.close_col = close_col
        self.back_looking = back_looking
        self.current_step = back_looking
        
        self.past_holdings = np.zeros(back_looking)
        self.past_nav = np.zeros(back_looking)

        assert len(data.shape) == 2, 'check input data is two-dimensional'
        assert data.shape[0] > 0, 'check data length'
        num_obervation = data.shape[0]
        num_features = data.shape[1]

        # first number: 0 = buy, 1 = sell, 2 = hold; second number: percent of holding as trade size
        #TODO: is there any better way of defining action space?
        self.action_space = gym.spaces.Box(low = np.array([0,0]), high = np.array([3,1]), dtype = np.float16) 

        #all features from input + current holdings + current NAV
        self.observation_space =gym.spaces.Box(low = MIN, high = MAX, shape = (back_looking,num_features+2))

    
    def _observe(self):
        """
        observe rows up to current_step - 1, basically saying you execute the trade for 
        the current_step using all the information available at the begining of the time
        """
        obs = np.zeros((back_looking,num_features+2))
        obs[:,0:self.num_features] = self.data[current_step - back_looking:current_step,:]
        obs[:,-2:-1] = self.past_holdings.reshape((back_looking,1))
        obs[:,-1:]   = self.past_nav.reshape((back_looking,1))
        return obs
    
    def _calculate_reward(holdings: int):
        """
        calculate gain from current holings
        TODO: consider discount, trade cost
        """
        return holdings*(data[current_step,self.close_col] - data[current_step,self.close_col])
        
    def step(self,action):
        """
        Returns
        -------
        observation

        reward: float

        episode_over: bool
        when you lose all your money 
        and go thru the data set?
        if you don't stop at some point: how to implement episode-based updates?

        info: dict
        diagnostic
        """

    
    def reset(self):
        ...
    def render(self, mode = 'human', close = False)
        ...