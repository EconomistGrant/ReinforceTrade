# ReinforceTrade
Author: Songhao LI, MIT Master of Finance\
Date: 2020/12/29\
A new project based on tensortrade

Why start this new project: 
1) practice RL / programming techniques
2) have your own modularized, modifiable RL project
3) learn to use some good frameworks (mostlikely TF) through the process
4) explore some common features / caveats of machine learning that might be important in later study / work / intervierws


Objectives:
1) Modular. Easy to implement / modify / upgrade 
2) Universal. Fits as many algorithms / data set formats as possible
3) Efficient. Explore techniques like GPU / multicore/ multithreads calculation

Talk is Cheap. Show me the code

https://neptune.ai/blog/the-best-tools-for-reinforcement-learning-in-python
https://medium.com/data-from-the-trenches/choosing-a-deep-reinforcement-learning-library-890fb0307092

candidates of calculation core:
TF-agents https://github.com/tensorflow/agents
Supports TF2.0+, not so easy to start

Stable baselines
- does not support TF2.0+


# 1/5 思考
第一个要考虑的点是environment
还是按照之前设计的思路 observation包括市场观察和当前持仓
action是交易

分两条路去做 一边研究gym这个environment 一边把这个DQN在本地跑通一次试试

刘神建议学习：
装饰器 https://www.jianshu.com/p/ee82b941772a
venv

# 1/8 Update
start working on environment definition
good reference: 
https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
https://github.com/notadamking/Stock-Trading-Environment/blob/master/env/StockTradingEnv.py 
https://gym.openai.com/docs/

# 1/12 Update
trading env看起来是写好了 还行吧170行代码。。。。。。。。。。
明天和fidelity开完会 下午debug一下 上一点真实数据 争取周末跑个模型试试看

# 3/27 
周末直接用torch搭一个dqn
就先做discrete action space + simple reward
然后用一个网络去predict best action
然后迭代网络

但是迭代要怎么迭代呢？
首先观察，用网络做决策，观察reward，加入buffer
在buffer里面的所有值，
  next_state_values用target_net 去算next states * actions 的 Q net 的最大值
  expected_state_action_values 就是上面的 next_state_values * GAMMA + reward
  loss是两者之差



周末跑通，有时间把action和reward抽象化
然后优化dqn
然后换别的算法/别的action reward方法

# 4/2
dqn搭起来了也简单能运行了
现在有两大问题：
I 模型比较粗糙
1. 模型本身：loss function，激活函数，epsilon-greedy还是直接用logistic正则化再采样，这些东西最好找到论文或者别人的项目看看
2. 超参数：evaluate_frequency, optimize_frequence, target_update_frequency, learning rate, batch_size

II 运行起来学不到东西
