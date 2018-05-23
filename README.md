# Deep-Q-learning-Cartpole-Problem
Description: This project is to train an agent to solve the CartPole problem in OpenAI Gym by implementing a Deep Q Network. The Gym package can be downloaded from gym.openai.com. 

Basic Idea: 
- Since CartPole has a continuous state space, it cannot use a table representation for the Q-function. Instead, I 
used a neural network to represent the Q-function. 
- Replace the default random agent by a Deep Q Network (DQN) agent. More precisely, I implemented Qlearning where the Q-function is represented by a neural network that takes as input a state and outputs a Q-value for each possible action. 
- Use TensorFlow to implement a neural network. 

Deep Q Network Configuration:
- Input layer of 4 nodes (corresponding to the 4 state features)
- Two hidden layers of 10 rectified linear units (fully connected)
- Output layer of 2 identity units (fully connected) that compute the Q-values of the two actions

