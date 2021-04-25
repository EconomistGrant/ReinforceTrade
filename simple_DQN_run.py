# -*- coding: utf-8 -*-
#%%
import sys
sys.path.append("..")
from core.Environment.env.simple_trade import TradingEnv
from core.Agent.DQNAgent import DQNAgent
import torch
import pandas as pd
import matplotlib.pyplot as plt

#%%
data =  pd.read_csv('data/AAPL.csv').values[:,1:5][:1000]

data = data/data[0] # normalize
env = TradingEnv(data = data, close_col = 3, back_looking = 30, rf = 0, initial_capital = 1)
agent = DQNAgent(env)
hist = agent.train(evaluate_frequency = 1)
# %%
