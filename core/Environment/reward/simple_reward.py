from ReinforceTrade.core.Environment.reward import RewardStrategy

class SimpleReward(RewardStrategy):
    def get_reward(self, environment):
       return environment.past_nav[-1] - environment.past_nav[-2]