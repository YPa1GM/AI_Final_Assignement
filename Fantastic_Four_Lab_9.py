#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
random.seed(10)
# --- Bandit ---
class BinaryBandit(object):
  def __init__(self):
    # N = number of arms
    self.N = 2
  def actions(self):
    result = []
    for i in range(0,self.N):#returns a list of possible actions in the bandit problem. In this case, there are two possible actions (arms), so the method returns a list containing the integers 0 and 1.
      result.append(i)
    return result
  def reward1(self, action):#reward1 that takes an action as input and returns a reward for that action. The reward is determined by a Bernoulli distribution with success probability p[action]
    p = [0.1, 0.2]
    rand = random.random()
    if rand < p[action]:
      return 1
    else:
      return 0
  def reward2(self, action):
    p = [0.7, 0.8]
    rand = random.random()
    if rand < p[action]:
      return 1
    else:
      return 0

def eGreedy_binary(myBandit, epsilon, max_iteration):
  # Initialization 
  Q = [0]*myBandit.N 
  count = [0]*myBandit.N
  r = 0
  R = []
  R_avg = [0]*1
  max_iter = max_iteration
  # Incremental Implementation
  for iter in range(1,max_iter):
    if random.random() > epsilon:
      action = Q.index(max(Q)) # Exploit/ Greed
    else:
      action = random.choice(myBandit.actions()) # Explore
    r = myBandit.reward1(action)
    R.append(r)
    count[action] = count[action]+1
    Q[action] = Q[action]+(r - Q[action])/count[action]
    R_avg.append(R_avg[iter-1] + (r-R_avg[iter-1])/iter)

  return Q, R_avg, R

#Initializing binary bandit problem,applying epsilon greedy algo to the problem with a fixed exploration rate of 0.2 and maximum 2000 iterations
random.seed(10)
myBandit = BinaryBandit()
Q, R_avg, R = eGreedy_binary(myBandit, 0.2, 1000)

import matplotlib.pyplot as plt
# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(R_avg)
ax1.title.set_text("Average rewards V/s Iteration")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(R)
ax2.title.set_text("Reward per iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Reward")

"""# n-Arm Bandit with unmodified ε-greedy"""

import random
# Creating the Bandit Env
class Bandit(object):
  def __init__(self, N):
    # N = number of arms
    self.N = N
    expRewards = [10]*N # [10,10,10,10], N=4
    self.expRewards = expRewards
  def actions(self):
    result = list(range(0,self.N))
    return result
  def reward(self, action):
    result = []
    
    for i in range(len(self.expRewards)):
      self.expRewards[i]+=random.gauss(0,0.1)

    result = self.expRewards[action]+random.gauss(0,0.01)
    return result

def eGreedy(myBandit, epsilon, max_iteration):
  # Initialization 
  Q = [0]*myBandit.N 
  count = [0]*myBandit.N
  epsilon = epsilon
  r = 0
  R = []
  R_avg = [0]
  max_iter = max_iteration
  # Incremental Implementation
  for iter in range(1,max_iter):
    if random.random() > epsilon:
      action = Q.index(max(Q)) # Exploit/ Greed
    else:
      action = random.choice(myBandit.actions()) # Explore
    r = myBandit.reward(action)
    R.append(r)
    count[action] = count[action]+1
    Q[action] = Q[action]+(r - Q[action])/count[action]
    R_avg.append(R_avg[iter-1] + (r-R_avg[iter-1])/iter)

  return Q, R_avg, R

random.seed(10)
myBandit = Bandit(10)
Q, R_avg, R = eGreedy(myBandit, 0.3, 10000)

print("Actual\tRecovered ")
for i,j in zip(myBandit.expRewards, Q):
    print(f"{i:.3f} \t {j:.3f}")

import matplotlib.pyplot as plt
# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(R_avg)
ax1.title.set_text("Average rewards V/s Iteration")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(R)
ax2.title.set_text("Reward per iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Reward")
fig.suptitle("Unmodified Epsilon Greedy Policy")

"""# n-Arm Bandit with modified ε-greedy"""

def eGreedy_modified(myBandit, epsilon, max_iteration, alpha):
  # Initialization 
  Q = [0]*myBandit.N 
  count = [0]*myBandit.N
  epsilon = epsilon
  r = 0
  R = []
  R_avg = [0]*1
  max_iter = max_iteration
  # Incremental Implementation
  for iter in range(1,max_iter):
    if random.random() > epsilon:
      action = Q.index(max(Q)) # Exploit/ Greed
    else:
      action = random.choice(myBandit.actions()) # Explore
    r = myBandit.reward(action)
    R.append(r)
    count[action] = count[action]+1
    Q[action] = Q[action]+ alpha*(r - Q[action])
    R_avg.append(R_avg[iter-1] + (r-R_avg[iter-1])/iter)

  return Q, R_avg, R

random.seed(10)
myBandit = Bandit(N=10)
Q, R_avg, R = eGreedy_modified(myBandit, 0.4, 10000, 0.01)

print("Actual\tRecovered ")
for i,j in zip(myBandit.expRewards, Q):
    print(f"{i:.3f} \t {j:.3f}")

import matplotlib.pyplot as plt
# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(R_avg)
ax1.title.set_text("Average rewards V/s Iteration")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Average Reward")
ax2.plot(R)
ax2.title.set_text("Reward per iteration")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Reward")
fig.suptitle("Modified Epsilon Greedy Policy")


# In[ ]:




