# ReinforceTrade
A modulized program that applies Reinforcement Learning to Quantitative Trading

Running is as simple as executing simple_DQN_run.py, that applies simple DQN to AAPL

# Modules
## Agent Module
Pytorch-based RL agents that calculate, receive environment feedback(rewards), and optimize parameters in training loop
API please refer to core/Agent/Agent.py

## Environment Module
OpenAI-Gym-based environments that describe the quantitative trading environments, generate observations, and calculate rewards.
Two sub-modules are core/Environment/action and core/Environment/reward