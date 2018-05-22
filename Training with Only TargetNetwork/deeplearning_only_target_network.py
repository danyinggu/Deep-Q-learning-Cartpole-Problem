# CS 486, A4, Q3
# Danying Gu, 20562708
# training only use target network
#Update the the target network after every 2 episode

import tensorflow as tf
import gym as gym
import random as rd
import numpy as np

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
# Target Network
TargetNetworkin_layer = tf.placeholder(tf.float32, [1, 4])

TargetNetworkweightFn1 = tf.Variable(tf.random_uniform([4, 10], -1, 1))
TargetNetworkweightFn2 = tf.Variable(tf.random_uniform([10, 10], -1, 1))

TargetNetworkb1 = tf.Variable(tf.random_uniform([1,10], -2, 2))
TargetNetworkb2 = tf.Variable(tf.random_uniform([1,10], -2, 2))

TargetNetworkweightFnOut = tf.Variable(tf.random_uniform([10, 2], -1, 1))
TargetNetworkb3 = tf.Variable(tf.random_uniform([1,2], -2, 2))

TargetNetworklayer1_out = tf.nn.relu(tf.matmul(TargetNetworkin_layer, TargetNetworkweightFn1) + TargetNetworkb1)
TargetNetworklayer2_out = tf.nn.relu(tf.matmul(TargetNetworklayer1_out, TargetNetworkweightFn2) + TargetNetworkb2)
TargetNetworkl_out = tf.matmul(TargetNetworklayer2_out, TargetNetworkweightFnOut) + TargetNetworkb3



# training function
target = tf.placeholder(tf.float32, [1, 2])
# loss
lossVal = tf.reduce_sum(tf.squared_difference(l_out, target))
forward_step = tf.train.AdagradOptimizer(0.1).minimize(lossVal)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

horizon_num = 500
episode_num = 1000
rewardLst = []
env = gym.make('CartPole-v0')
for i_episode in range(episode_num):
    observation = env.reset()
    for t in range(horizon_num):

        # Choose selectedAction
        observationInput =np.reshape(observation, (1, 4))

        QFunction = sess.run(l_out, feed_dict = {in_layer:observationInput})
        resultedQFunction = np.copy(QFunction)
        QFunction = QFunction[0]
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

        # get new state
        nextObservation, nextReward, done, info = env.step(selectedAction)

        nextObservationInput = np.reshape(nextObservation, (1,4))
        nextQFunction = sess.run(TargetNetworkl_out, feed_dict={TargetNetworkin_layer: nextObservationInput })

        # update the target Q value according to it's done or not
        if done is True:
            resultedQFunction[0, selectedAction] = nextReward
        else:
            resultedQFunction[0, selectedAction] = nextReward + discount * np.max(nextQFunction)

        # put the current Q value and target Q value into the training step as moving forward
        sess.run(forward_step, feed_dict={in_layer: observationInput, target: resultedQFunction})

        if done is True:
            totalReward = 0
            #calculate the discounted total reward
            for timestep in range(t+1):
                totalReward += discount**timestep
            rewardLst.append(totalReward)
            print str(i_episode+1) + " " + str(t+1)
            break

        observation = nextObservation
    # update the network once two episodes
    if i_episode % 2 == 1:
        sess.run(TargetNetworkweightFn1.assign(weightFn1))
        sess.run(TargetNetworkweightFn2.assign(weightFn2))
        sess.run(TargetNetworkweightFnOut.assign(weightFnOut))
        sess.run(TargetNetworkb1.assign(b1))
        sess.run(TargetNetworkb2.assign(b2))
        sess.run(TargetNetworkb3.assign(bout))

print(rewardLst)
sess.close()
