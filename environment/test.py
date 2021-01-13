# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:37:02 2021

@author: a
"""
import numpy as np
from gym_trading.envs.trading_env import TradingEnv

class RandomAgent(object):
    def __init__(self,action_space):
        self.action_space = action_space
        
    def act(self):
        return self.action_space.sample()


data = np.array([[1,0.1,0.01],
                 [2,0.2,0.02],
                 [3,0.3,0.03],
                 [4,0.4,0.04],
                 [5,0.5,0.05],
                 [6,0.6,0.06]])
env = TradingEnv(data,0.01,10000,0,2)
agent = RandomAgent(env.action_space)
obs = env.reset()
action = agent.act()

env.step(action)
action = agent.act()
action = agent.act()
action = agent.act()