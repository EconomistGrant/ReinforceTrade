# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:21:49 2021

@author: a
"""
#%% Import 
import base64
import imageio
import IPython
import matplotlib.pyplot as plt
import os
import reverb
import tempfile
import PIL.Image

import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

tempdir = tempfile.gettempdir()

from tf_agents.environments import utils
import pandas as pd
#%% hyperparameters
num_iterations = 100 # @param {type:"integer"}

initial_collect_steps = 50 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10 # @param {type:"integer"}

batch_size = 8 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 5000 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 5 # @param {type:"integer"}

policy_save_interval = 50 # @param {type:"integer"}

#%% environment
from pyenv_single_action import TradingEnv
from tf_agents.environments import tf_py_environment

data = pd.read_csv('environment/GSPC.csv')[['Date','Open','Close','High','Low']]
data['Date'] = pd.to_datetime(data['Date'])
data_input = data.values[:,1:]
#data_input = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

env = TradingEnv(data = data_input, rf = 0, back_looking = 4)
utils.validate_py_environment(env, episodes=5)

train_env = tf_py_environment.TFPyEnvironment(env)

#%% Use GPU
strategy = strategy_utils.get_strategy(tpu = False, use_gpu = True)

#%% Agent (A/C Nets)
observation_spec, action_spec, time_step_spec = (
      spec_utils.get_tensor_specs(train_env))

with strategy.scope():
  critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')
  
with strategy.scope():
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=(
          tanh_normal_projection_network.TanhNormalProjectionNetwork))
  

#%% Actors




















































