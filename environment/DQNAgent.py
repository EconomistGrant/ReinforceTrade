from collections import namedtuple
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.optim as optim

from simple_trade import TradingEnv


torch.set_default_tensor_type(torch.DoubleTensor)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):
    def __init__(self,capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)



class DQN(nn.Module):
    def __init__(self,env:TradingEnv):
        super(DQN,self).__init__()
        self.in_channels = env.observation_space.shape[0]  #back looking window. Could be 30?
        self.length = env.observation_space.shape[1]       #normally should be 6: open close high low holdings nav
        self.num_actions = env.action_space.n

        self.conv1 = nn.Conv1d(self.in_channels,32,3)      #should be num_batches * 32 * 4
        self.conv2 = nn.Conv1d(32,8,3)                     #should be num_batches * 8 * 2
        self.fc    = nn.Linear(8 * (self.length - 4), env.action_space.n)

    def forward(self,x):
        
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = x.view(-1,self.num_flat_features(x))
        x = self.fc(x)

        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class DQNAgent(object):
    def __init__(self, 
                 env:'TradingEnv',
                 net:'DQN' = None,
                 batch_size = 1000):
        self.env = env
        self.policy_net = net or DQN(env)
        
        self.target_net = type(self.policy_net)(self.env)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.buffer = ReplayBuffer(1000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.batch_size = batch_size
        self.gamma = 0.999 #discounting
        self.eps_start = 0.9 #explore
        self.eps_end = 0.05
        self.eps_decay = 2000
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        
    def act(self,obs):
        #this function is greedy, epsilon implemented in train
        input = torch.from_numpy(obs).to(self.device)
        with torch.no_grad():
            return self.policy_net(input).max(1)[1] 
        
    def optimize(self):
        if len(self.buffer) < self.batch_size: return
        
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])        
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        
        
    def train(self):
        for i_episode in range(100):
            done = False
            obs = np.expand_dims(env.reset(),axis = 0)
            state = torch.from_numpy(obs).to(self.device)
            while not done:
                #TODO: epsilon_greedy?
                action = self.act(state)
                action_value = action.cpu().numpy()[0] #might be slow?
                next_obs,reward,done,_ = env.step(action_value)

                reward = torch.tensor([reward],device = self.device)
                if not done:
                    next_obs = np.expand_dims(next_obs,axis = 0)
                    next_state = torch.from_numpy(next_obs).to(self.device)
                else:
                    next_state = None

                self.buffer.push(state,action,next_state,reward)

                state = next_state

                #TODO: optimize_frequency
                self.optimize()

                #TODO: target_optimize_frequency
                self.target_net.load_state_dict(self.policy_net.state_dict())

                #TODO: evaluate frequence
                self.evaluate

        return
            
    
    def evaluate():
        raise NotImplementedError

    def save(self,path):
        raise NotImplementedError

if __name__ == '__main__':
    # DQN-test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data =  pd.read_csv('GSPC.csv').values[:,1:5]
    env = TradingEnv(data = data, close_col = 3, back_looking = 30, rf = 0.06/253, initial_capital = 10000)
    dqn = DQN(env).to(device)
    
    obs = np.expand_dims(env.reset(), axis = 0) 
    input = torch.from_numpy(obs).to(device)
    output = dqn(input)
    action = output.max(1)[1]
    print(action)

    # Agent-test
    agent = DQNAgent(env)