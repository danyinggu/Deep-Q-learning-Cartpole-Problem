# Danying Gu, A4 ,Q3
# train without target network or experience buffer

import tensorflow as tf
import gym as gym
import numpy as np
import random as rd

epsilon = 0.05
discount = 0.99


#input layer, 4 state features
in_layer = tf.placeholder(tf.float32, [1, 4])
#weight function for inner layer 1, 10 units for each states
weightFn1 = tf.Variable(tf.random_uniform([4, 10], -1, 1))
#weight function for inner layer 2, 10 units for each unit
weightFn2 = tf.Variable(tf.random_uniform([10, 10], -1, 1))

#bias functions for 2 inner layers
b1 = tf.Variable(tf.random_uniform([1,10], -2, 2)) #one bias for each units
b2 = tf.Variable(tf.random_uniform([1,10], -2, 2)) #one bias for each units

#weight function and bias for output layer
weightFnOut = tf.Variable(tf.random_uniform([10, 2], -1, 1))
bout = tf.Variable(tf.random_uniform([1,2], -2, 2))

#activation function for each layer to generate the output
l1_out = tf.nn.relu(tf.matmul(in_layer, weightFn1) + b1)
l2_out = tf.nn.relu(tf.matmul(l1_out,weightFn2) + b2)
# 2 output units
l_out = tf.matmul(l2_out,weightFnOut) + bout

# about the target funciton. loss and the forward_step to minimize the loss
target = tf.placeholder(tf.float32, [1, 2])
lossVal = tf.reduce_sum(tf.squared_difference(target, l_out))
forward_step = tf.train.AdagradOptimizer(0.1).minimize(lossVal)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

rewardLst=[]
horizon_num = 500
episode_num = 1000
env = gym.make('CartPole-v0')
for i_episode in range(episode_num):
    observation = env.reset()
    for t in range(horizon_num):
        #env.render()
        #print ("a")

        # feed the observation data to placeholder in_layer
        # and take the input into neural network to generate the output
        inputObservation = np.reshape(observation, (1,4)) # reshape the observation to form of input layer
        QFunction = sess.run(l_out, feed_dict = {in_layer:inputObservation})
        targetQFunction = np.copy(QFunction)
        QFunction = QFunction[0]

        #select an action
        actionprob = rd.random()
        # if >= epsilon, then choose the action with maximum Q value in the output
        if actionprob >= epsilon:
            if QFunction[0] > QFunction[1]:
                selectedAction = 0
            elif QFunction[1] > QFunction[0]:
                selectedAction = 1
            else:
                selectedAction = env.action_space.sample()
        # if < epsilon, then choose a random action in the action space
        else:
            selectedAction = env.action_space.sample()


        # take a step
        nextObservation, nextReward, done, info = env.step(selectedAction)
        # reshape the next observation(state) to the form of input layer
        inputNextObservation = np.reshape(nextObservation, (1,4))
        nextQFunction = sess.run(l_out, feed_dict = {in_layer: inputNextObservation})
        # if it is fallen, return the reward directly
        if done == True:
            targetQFunction[0,selectedAction] = nextReward
        else:
            targetQFunction[0,selectedAction] = (discount * (np.max(nextQFunction))) + nextReward

        # do the forward training step
        sess.run(forward_step, feed_dict = {in_layer:inputObservation, target: targetQFunction})

        if done:
            totalReward = 0
            # calculate the discounted total reward
            for s in range(t+1):
                totalReward += discount**s
            rewardLst.append(totalReward)
            print str(i_episode+1) + " " + str(t+1)
            break
print(rewardLst)
sess.close()
