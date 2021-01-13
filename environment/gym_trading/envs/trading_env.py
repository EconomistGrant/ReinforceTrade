import random
import gym
import numpy as np

MAX = 10000
MIN = -10000
class TradingEnv(gym.Env):
    """
    A trading environment set up using OpenAI Gym
    Data are stored in numpy.array format
    
    
    Attributes
    ----------
    data
    current_step
    
    holding
    cash
    nav

    past_holdings: 
    past_nav
    """
    def __init__(self,
                 data: np.array, 
                 rf: float,
                 initial_capital: float = 10000.0,
                 close_col: int = 1,
                 back_looking :int = 1,):
        super(TradingEnv,self).__init__()

        #TODO: standardize input
        
        self.data = data
        self.rf = rf
        self.close_col = close_col
        self.back_looking = back_looking
        self.initial_capital = initial_capital

        assert len(data.shape) == 2, 'check input data is two-dimensional'
        assert data.shape[0] > 0, 'check data length'
        self.num_obervation = data.shape[0]
        self.num_features = data.shape[1]

        # first number: 0 = sell, 1 = hold, 2 = buy; second number: percent of nav as trade size
        #TODO: is there any better way of defining action space?
        self.action_space = gym.spaces.Box(low = np.array([0.0,0.0]), high = np.array([3.0,1.0]), dtype = np.float64) 

        #all features from input + current holdings + current NAV
        self.observation_space =gym.spaces.Box(low = MIN, high = MAX, shape = (self.back_looking,self.num_features+2))

    
    def _observe(self):
        """
        observe rows up to current_step - 1, basically saying you execute the trade for 
        the current_step using all the information available at the begining of the time
        price at current_step - 1 is the trade price, hold until the end of current_step
        """
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
        

        

    def _calculate_reward(self):
        """
        calculate gain from current holings
        TODO: consider discount, trade cost
        """
        return self.holding*(self.data[self.current_step,self.close_col] - self.data[self.current_step-1,self.close_col]) + self.cash*self.rf
    

    def step(self,action):
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

        self._execute_trade(action)
        
        reward = self._calculate_reward()
       
        
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

        self.current_step += 1
        obs = self._observe()

        done = self.nav <= 0 or self.current_step >= self.data.shape[0]
        
        return obs, reward, done, {} 


    
    def reset(self):
        self.past_holdings = np.zeros(self.back_looking)
        self.past_nav = np.zeros(self.back_looking)

        self.holding = 0
        self.cash = self.initial_capital
        self.nav = self.initial_capital
        self.current_step = self.back_looking

        return self._observe()
        
    def render(self, mode = 'human', close = False):
        print('nav:',self.nav)