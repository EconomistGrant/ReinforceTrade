# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:37:02 2021

@author: a
"""
import numpy as np
import pandas as pd
from trading_env import TradingEnv

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

data2 = np.ones(shape = (253,4))

data3 = pd.read_csv('GSPC.csv').values[:,1:5]

env = TradingEnv(data = data3, close_col = 3, back_looking = 2, rf = 0.06/253, initial_capital = 10000)
agent = RandomAgent(env.action_space)

steps = []
nav = []
for i in range(1,100):
    obs = env.reset()
    done = False
    while not done:
        action = agent.act()
        _,_,done,_ = env.step(action)
    
    steps.append(env.current_step)
    nav.append(env.nav)
