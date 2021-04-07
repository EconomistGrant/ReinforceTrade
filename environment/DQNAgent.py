from collections import namedtuple
import matplotlib.pyplot as plt
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
        self.in_channels = env.observation_space.shape[0]         #back looking window. Could be 30?
        self.num_variables = env.observation_space.shape[1]       #normally should be 6: open high low close holdings nav
        self.num_actions = env.action_space.n

        self.conv1 = nn.Conv1d(self.in_channels,32,3)      #should be num_batches * 32 * 4
        self.conv2 = nn.Conv1d(32,8,3)                     #should be num_batches * 8 * 2
        self.fc    = nn.Linear(8 * (self.num_variables - 4), env.action_space.n)

    def forward(self,x):
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1,self.num_flat_features(x))
        x = self.fc(x)

        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class simple_DQN(nn.Module):
    """
    No CNN: fully connecting everything
    """
    def __init__(self, env:TradingEnv):
        super(simple_DQN, self).__init__()
        self.shape = env.observation_space.shape
        self.num_features = self.shape[0] * self.shape[1]
        self.fc1 = nn.Linear(self.num_features,64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16,env.action_space.n)
        
    def forward(self,x):
        x = x.view(-1,self.num_features)
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DQNAgent(object):
    def __init__(self, 
                 env:'TradingEnv',
                 net:'DQN' = None,
                 batch_size = 32,
                 learning_rate = 1e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.policy_net = net or simple_DQN(env)
        self.policy_net = self.policy_net.to(self.device)
        
        self.target_net = type(self.policy_net)(self.env).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.buffer = ReplayBuffer(200)
        self.batch_size = batch_size
        self.gamma = 0.9999 #discounting
        self.eps_start = 0.9 #explore
        self.eps_end = 0.05
        self.eps_decay_steps = 2000
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = learning_rate)
        
        
    def act(self,state,eps):
        """
        epsilon-greedy action
        """
        if random.random() < eps:
            action_value = self.env.action_space.sample()
            return torch.tensor(action_value).to(self.device).reshape(1)
        with torch.no_grad():
            return self.policy_net(state).max(1)[1] 
    
    def act_sm(self,state,eps):
        """
        soft-max style sampling actions
        """
        if random.random() < eps:
            action_value = self.env.action_space.sample()
            return torch.tensor(action_value).to(self.device).reshape(1)
        with torch.no_grad():
            raw = self.policy_net(state) #todo: are you sure about this?
            return torch.multinomial(torch.exp(raw), 1)[0]
        
    def optimize(self):
        if len(self.buffer) < self.batch_size: return
        
        #sample from memory
        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])        
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        action_batch = action_batch.reshape(action_batch.shape[0],-1)
        reward_batch = torch.cat(batch.reward)
        
        # Q_table: num_batches * action_space_size
        Q_table = self.policy_net(state_batch)
        
        # Select the actions taken (recorded in buffer)
        # The original action taken is by epsilon-greedy process with the net at that time
        state_action_values = Q_table.gather(1, action_batch)
        
        
        next_state_values = torch.zeros(self.batch_size, device = self.device)
        

        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # state_action_values: using policy net, calculate the values of the Q(s_t, a_t)
        # exptected_state_action_values: using target net, calculate the values of r(s_t, a_t) + V(s_t+1)
        
        loss = self.loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()
        return loss
        
        
    def train(self, 
              n_episode = 500, 
              evaluate_frequency = 10, #every x episode
              optimize_frequency = 10,  #every x steps, sample from buffer, optimize
              target_update_frequency = 2000, #every x steps, update target net
              ):
        step = 0
        eps = self.eps_start
        nav_hist = []
        for i_episode in range(n_episode):
            done = False
            obs = np.expand_dims(env.reset(),axis = 0)
            state = torch.from_numpy(obs).to(self.device)
            while not done:
                step += 1
                if eps > self.eps_end:
                    eps = self.eps_start - (self.eps_start - self.eps_end)*step/self.eps_decay_steps
                #or stochastic decision: exponential-normalize action vector
                
                #generate action
                
                #--- epsilon_greedy ---
                action = self.act(state,eps)
                action_value = action.cpu().numpy()[0]
                
                #interact with environment
                next_obs,reward,done,_ = env.step(action_value)
                
                if not done:
                    next_obs = np.expand_dims(next_obs,axis = 0)
                    next_state = torch.from_numpy(next_obs).to(self.device)
                else:
                    next_state = None
                
                #record into buffer
                reward = torch.tensor([reward],device = self.device)
                self.buffer.push(state,action,next_state,reward)

                
                if step % optimize_frequency == 0:
                    loss = self.optimize()
                    
                #target network update
                if step % target_update_frequency == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    
                #go to next step:    
                state = next_state
                
            #evaluate policy
            if i_episode % evaluate_frequency == 0:
                print("----------------------------")
                print("Episode: " + str(i_episode))
                print("loss ")
                print(loss)
                nav_hist.append(self.evaluate())
            
        return nav_hist
            
    
    def evaluate(self):
        #TODO: run multiple times and take average
        navs = []
        for i in range(5):
            done = False
            nav = []
            actions = []
            obs = np.expand_dims(self.env.reset(),axis = 0)
            state = torch.from_numpy(obs).to(self.device)
            while not done:
                action = self.act_sm(state,0.05)
                action_value = action.cpu().numpy()[0] #might be slow?
                _,_,done,_ = self.env.step(action_value)
                nav.append(self.env.nav)
                actions.append(action_value)
            #plt.plot(nav);plt.show();print(nav[-1])
            #plt.plot(actions);plt.show()
            navs.append(nav[-1])
        
        avrg_nav = np.mean(navs)
        print("Evaluate result: " + str(avrg_nav))
        return avrg_nav
    
    def evaluate_plot(self):
        #TODO: run multiple times and take average
        navs = []
        done = False
        nav = []
        actions = []
        obs = np.expand_dims(self.env.reset(),axis = 0)
        state = torch.from_numpy(obs).to(self.device)
        while not done:
            action = self.act_sm(state,0.05)
            action_value = action.cpu().numpy()[0] #might be slow?
            _,_,done,_ = self.env.step(action_value)
            nav.append(self.env.nav)
            actions.append(action_value) 
        plt.plot(nav)
        plt.plot(actions)
        return
    
    def save(self,path):
        raise NotImplementedError

if __name__ == '__main__':
    # DQN-test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data =  pd.read_csv('AAPL.csv').values[:,1:5]
    data = data/data[0] # normalize
    env = TradingEnv(data = data, close_col = 3, back_looking = 30, rf = 0.06/253, initial_capital = 10000)
    dqn = simple_DQN(env).to(device)
    
    obs = np.expand_dims(env.reset(), axis = 0) 
    input = torch.from_numpy(obs).to(device)
    output = dqn(input)
    action = output.max(1)[1]
    print(action)
    """
    # Agent-test
    data =  pd.read_csv('AAPL.csv').values[:,1:5][:1000]
    data = data/data[0] # normalize
    env = TradingEnv(data = data, close_col = 3, back_looking = 30, rf = 0, initial_capital = 1)
    agent = DQNAgent(env)
    hist = agent.train()
    
