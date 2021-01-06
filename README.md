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

