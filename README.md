# Deep-Q-learning-Cartpole-Problem
1. Description:
  - This project is to train an agent to solve the CartPole problem in OpenAI Gym by implementing a Deep Q Network. The Gym package can be downloaded from gym.openai.com. 

2. Basic Idea: 
  - Since CartPole has a continuous state space, it cannot use a table representation for the Q-function. Instead, I used a neural network to represent the Q-function. 
- Replace the default random agent by a Deep Q Network (DQN) agent, where I implemented Q-learning where the Q-function is represented by a neural network that takes as input a state and outputs a Q-value for each possible action. 
- Use TensorFlow to implement the neural network. 
- Learning Parameters:
  - Discount rate: 0.99
  - Exploration strategy: epsilon-greedy with epsilon=0.05
  - Learing Rate=0.1, using the adagradOptimizer
  - Train for a maximum of 1000 episodes

3. Deep Q Network Configuration:
- A 4-node Input layer, which corresponds to the feature of 4 states
- Two fully connected hidden layers of 10 rectified linear units
- A 2-units Output layer (fully connected) where each identity unit corresponds to the Q-values of the two actions

4. Training with 4 scenarios: 
- Q-learning without experience replay and no target network)
- Q-learning with only experience replay: use a replay buffer of size 1000 and replay a mini-batch of size 50 after each new experience. 
- Q-learning with only a target network: update the the target network after every 2 episodes.
- Q-learning with both experience replay and a target network, with same methods above.

5. Result:
- Produce a graph that shows the discounted total reward (y-axis) earned in each training episode as a function of the number of training episodes for each scenario.
- The graph result for scenario can be find under the folder for each scenario.
- A Brief Discussion: Since the deep Q neural network is non-linear, it is not grantee that the reward will be converge to some optimal value after many steps. 
    - In the first scenario, the discounted reward is not that high, and fluctuate much even after large number of episodes. However, when only added experience replay, the reward values fluctuate in first 300 episodes, and get close to a higher value, and vibrate less. 
    - Also, when added target network, the discounted total reward will get closed to a high value and stay more stable after 700 steps. (Although in my implementation, it seems that experience replay gets to the higher value faster and more stable than only target network method). 
    - Finally, with both target network and experience replay, the discount total reward of the training can get to the optimal value quickly (after about 200 steps) and converge on that result, with fluctuation occasionally. This result means that both experience replay buffer method and the target network method are useful in the deep Q learning for the rewards to converge and quickly get to the high value.


