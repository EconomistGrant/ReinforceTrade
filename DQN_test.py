# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 18:05:45 2021

@author: a
"""

#%% import
from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import pandas as pd

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.environments import utils

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.compat.v1.enable_v2_behavior()


#%% load trading environment
from pyenv_single_action import TradingEnv
data = pd.read_csv('environment/GSPC.csv')[['Date','Open','Close','High','Low']]
data['Date'] = pd.to_datetime(data['Date'])
data_input = data.values[:,1:]
#data_input = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

env = TradingEnv(data = data_input, rf = 0.05/200, back_looking = 4)
utils.validate_py_environment(env, episodes=5)

train_env = tf_py_environment.TFPyEnvironment(env)
#%% Hyperparameters
num_iterations = 100 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"} 
collect_steps_per_iteration = 100  # @param {type:"integer"}

replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 4  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}

log_interval = 5  # @param {type:"integer"}
num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 10  # @param {type:"integer"}\
    
#%% Agent
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(100,50))

target_q_network = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=(100,50))

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    target_q_network = target_q_network,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=tf.Variable(0))

agent.initialize()

#%% Policies
eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())


#%% metrics
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

'''
for _ in range(5):
    print(compute_avg_return(train_env, agent.policy, num_eval_episodes))
'''


'''210209'''
#%% Replay Buffer
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

#%% Data Collection
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)

iterator = iter(dataset)

#%% Training
#%%% Params
agent.train = common.function(agent.train)
# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(train_env, agent.policy, num_eval_episodes)

returns = [avg_return]
#%%% Run
for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss
  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(train_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
C:\Users\a\Desktop\work\2021_Spring\6.006 Algo\ass\ass0\tests.py    returns.append(avg_return)

#%% Random Run
for _ in range(num_iterations):
  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_data(train_env, random_policy, replay_buffer, collect_steps_per_iteration)
  # Sample a batch of data from the buffer and update the agent's network.
  step = agent.train_step_counter.numpy()
  print(step)
  
  if step % eval_interval == 0:
    avg_return = compute_avg_return(train_env,random_policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)
    
    

    









