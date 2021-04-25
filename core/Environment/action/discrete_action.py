from ReinforceTrade.core.Environment.action import ActionStrategy
import gym

class DiscreteAction(ActionStrategy):
    def __init__(self,num_actions):
        self.A = num_actions
        self.action_space = gym.spaces.Discrete(num_actions)
    
    def get_trade(self,action):
        """ return raw trade size between [-1,1]"""
        half_A = int(self.A / 2)
        trade = action/(half_A) - 1
        return trade