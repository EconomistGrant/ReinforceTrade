import random
import gym
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import BoundedArraySpec
from tf_agents.trajectories.time_step import StepType
from tf_agents.trajectories.time_step import TimeStep


MAX = 10000
MIN = -10000
class TradingEnv(py_environment.PyEnvironment):
    """
    A trading environment set up using OpenAI Gym
    Data are stored in numpy.array format
    
    
    Inputs
    ----------
    data:            np.array of shape (num_observation,num_features)
    close_col:       the column index for the close row in data
    back_looking:    length of observation window feeding to the agent
    rf:              risk-free rate
    initial_capital: initial capital to start learning

    Attributes
    ----------   
    action_space:      first number: 0 = sell, 1 = hold, 2 = buy
                       second number: percent of nav as trade size
    
    observation_space: inputs features + past holdings + past NAV, all having the length of back_looking
    current_step:      the timestep/row that the agent is currently at. Agent will observe up to (current_step - 1),
                       and trade at the end of (current_step - 1), hold until the end of current_step
    
    nav,cash,holdings: current portfolio/cash/equity value

    past_holdings, past_nav: portfolio history of length back_looking to be fed into the observation
    num_observation,num_features: data shape
    """
    def __init__(self,
                 data: np.array, 
                 close_col: int = 1,
                 back_looking :int = 2,
                 rf: float = 0.01,
                 initial_capital: float = 10000.0,
                 discount: float = 1.0
                 ):
        super(TradingEnv,self).__init__()

        #TODO: standardize input
        self.data = data
        self.rf = rf
        self.close_col = close_col
        self.back_looking = back_looking
        self.initial_capital = initial_capital

        #TODO: discount
        self._discount = np.asarray(discount, dtype=np.float32)

        assert len(data.shape) == 2, 'check input data is two-dimensional'
        assert data.shape[0] > 0, 'check data length'
        self.num_obervation = data.shape[0]
        self.num_features = data.shape[1]

        #all features from input + current holdings + current NAV
        self.observation_space =gym.spaces.Box(low = MIN, high = MAX, shape = (self.back_looking,self.num_features+2))

    def action_spec(self):
        return BoundedArraySpec(shape = (2,), dtype = np.float32, minimum=[0.0,0.0], maximum=[3.0,1.0], name = 'action')
    
    def observation_spec(self):
        return BoundedArraySpec(shape = (self.back_looking,self.num_features+2), dtype = np.float64, minimum=-999999, maximum=999999)

    def _observe(self):
        """observe rows up to current_step - 1"""
        obs = np.zeros((self.back_looking,self.num_features+2))
        obs[:,0:self.num_features] = self.data[self.current_step - self.back_looking:self.current_step,:]
        obs[:,-2:-1] = self.past_holdings.reshape((self.back_looking,1))
        obs[:,-1:]   = self.past_nav.reshape((self.back_looking,1))
        return obs
    
    def _validate_trade(self,cash,exposure):
        if cash < 0:
            return False
        if abs(exposure) > self.nav:
            return False
        return True
    
    def _execute_trade(self,action):
        prev_holding = self.holding
        prev_cash = self.cash

        trade_price =  self.data[self.current_step-1,self.close_col]

        trade_type = action[0]
        trade_amount = (action[1] * self.nav)/trade_price

        if trade_type < 1:
            # sell, limit FOR NOW that net shorting values < nav
            cash = prev_cash + trade_amount * trade_price
            holding = prev_holding - trade_amount
        
        elif trade_type < 2:
            # buy, limit cash >= 0
            cash = prev_cash - trade_amount * trade_price
            holding = prev_holding + trade_amount

        else:
            cash = prev_cash
            holding = prev_holding
        
        #validate trade
        if self._validate_trade(cash, holding*trade_price):
            self.cash = cash
            self.holding = holding

        #update holdings log
        temp_past_holdings = self.past_holdings
        self.past_holdings[0:-1] = temp_past_holdings[1:]
        self.past_holdings[-1] = self.holding

        #update nav log
        self.cash = self.cash* (1+self.rf)
        self.nav = self.holding * self.data[self.current_step,self.close_col] + self.cash
        temp_past_nav = self.past_nav
        self.past_nav[0:-1] = temp_past_nav[1:]
        self.past_nav[-1] = self.nav
        
        

    def _calculate_reward(self):
        """
        calculate gain from current holings
        TODO: consider discount, trade cost
        """
        return self.past_nav[-1] - self.past_nav[-2]
        #return self.holding*(self.data[self.current_step,self.close_col] - self.data[self.current_step-1,self.close_col]) + self.cash*self.rf
    

    def _step(self,action):
        """
        act -> reward -> update nav -> next observation
        
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
        
        if self.current_step >= self.data.shape[0]:
            return TimeStep(StepType.LAST, np.float32(0), self._discount, np.zeros(shape = (self.back_looking,self.num_features+2), dtype = np.float64))
       
        self._execute_trade(action)
        
        reward = self._calculate_reward()
        
        
        self.current_step += 1
        obs = self._observe()

        done = self.nav <= 0 or self.current_step >= self.data.shape[0]
        
        step_type = StepType.MID
        if self.current_step == self.back_looking:
            step_type = StepType.FIRST
        elif done:
            step_type = StepType.LAST
        
        return TimeStep(step_type, reward.astype('float32'), self._discount, obs)
            

    def _reset(self):
        self.past_holdings = np.zeros(self.back_looking)
        self.past_nav = np.ones(self.back_looking) * self.initial_capital

        self.holding = 0
        self.cash = self.initial_capital
        self.nav = self.initial_capital
        
        self.current_step = self.back_looking

        obs = self._observe()

        return TimeStep(StepType.FIRST,  np.asarray(0.0, dtype=np.float32), self._discount, obs)
        
