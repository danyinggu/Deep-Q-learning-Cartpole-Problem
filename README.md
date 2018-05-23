# Deep-Q-learning-Cartpole-Problem
# Description:
- This project is to train an agent to solve the CartPole problem in OpenAI Gym by implementing a Deep Q Network. The Gym package can be downloaded from gym.openai.com. 

# Basic Idea: 
- Since CartPole has a continuous state space, it cannot use a table representation for the Q-function. Instead, I used a neural network to represent the Q-function. 
- Replace the default random agent by a Deep Q Network (DQN) agent, where I implemented Q-learning where the Q-function is represented by a neural network that takes as input a state and outputs a Q-value for each possible action. 
- Use TensorFlow to implement the neural network. 

# Deep Q Network Configuration:
- A 4-node Input layer, which corresponds to the feature of 4 states
- Two fully connected hidden layers of 10 rectified linear units
- A 2-units Output layer (fully connected) where each identity unit corresponds to the Q-values of the two actions

# Training with 4 scenarios: 
- Q-learning without experience replay and no target network)
- Q-learning with only experience replay: use a replay buffer of size 1000 and replay a mini-batch of size 50 after each new experience. 
- Q-learning with only a target network: update the the target network after every 2 episodes.
- Q-learning with both experience replay and a target network, with same methods above.

# Result:
- Produce a graph that shows the discounted total reward (y-axis) earned in each training episode as a function of the number of training episodes for each scenario.


