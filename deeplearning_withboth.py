#CS 486, A4, Q3
# Danying GU, 20562708
# train using both target network and experience replay buffer
import tensorflow as tf
import gym as gym
import numpy as np
import random as rd

epsilon = 0.05
discount = 0.99

#input layer, 4 state features
in_layer = tf.placeholder(tf.float32, [None, 4])
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

#information of the buffer
buffNum = 0
buffSize = 1000
miniBatchSize = 50
expBuffer = []

# Target Network
TargetNetworkin_layer = tf.placeholder(tf.float32, [None, 4])

TargetNetworkweightFn1 = tf.Variable(tf.random_uniform([4, 10], -1, 1))
TargetNetworkweightFn2 = tf.Variable(tf.random_uniform([10, 10], -1, 1))

TargetNetworkb1 = tf.Variable(tf.random_uniform([1,10], -2, 2))
TargetNetworkb2 = tf.Variable(tf.random_uniform([1,10], -2, 2))

TargetNetworkweightFnOut = tf.Variable(tf.random_uniform([10, 2], -1, 1))
TargetNetworkbout = tf.Variable(tf.random_uniform([1,2], -2, 2))

TargetNetworklayer1_out = tf.nn.relu(tf.matmul(TargetNetworkin_layer, TargetNetworkweightFn1) + TargetNetworkb1)
TargetNetworklayer2_out = tf.nn.relu(tf.matmul(TargetNetworklayer1_out, TargetNetworkweightFn2) + TargetNetworkb2)
TargetNetworkl_out = tf.matmul(TargetNetworklayer2_out, TargetNetworkweightFnOut) + TargetNetworkbout



# training function
target = tf.placeholder(tf.float32, [None, 2])
# loss
lossVal = tf.reduce_sum(tf.squared_difference(target, l_out))
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
        newOb = np.copy(observation)
        QFunction = sess.run(l_out, feed_dict={in_layer: observationInput})[0]
        resultedQFunction = np.copy(QFunction)
        actionprob = rd.random()
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
        # get next state
        nextObservation, nextReward, done, info = env.step(selectedAction)
        newNextOb = np.copy(nextObservation)

        # make an experience item for the buffer, put current state and next state inside
        experience = [newOb, selectedAction, newNextOb, nextReward, done, info]
        # if the buffernumber is less than 1000, append the experience directly
        if buffNum < buffSize:
            expBuffer.append(experience)
        # if the buffer number is already 1000, replace the oldest one with the new one
        else:
            expBuffer[buffNum % buffSize] = experience
        buffNum = buffNum + 1

        # once the buffer number is greater than the miniBatch size (50)
        # do training in the miniBatch
        if buffNum >= miniBatchSize:
            # randomly select 50 experiences from the buffer into the miniBatch
            miniBatch = rd.sample(expBuffer, 50)
            nextVal = []
            observedVal = []
            # append the current obseravation value and next obseravation value list
            for exp in miniBatch:
                observedVal.append(exp[0])
                nextVal.append(exp[2])

            # calculate the current Q value and the next Q value (based on the target network)
            observedQVal = sess.run(l_out, feed_dict={in_layer: observedVal})
            resultedQFunction = np.copy(observedQVal)
            # next Q value based on the target network
            nextQVal = sess.run(TargetNetworkl_out, feed_dict={TargetNetworkin_layer: nextVal})
            expInd = 0
            while expInd < miniBatchSize:
                buffAction = miniBatch[expInd][1]
                buffNextReward = miniBatch[expInd][3]
                buffDone = miniBatch[expInd][4]
                # calculate the target Q value
                if buffDone is True:
                    resultedQFunction[expInd, buffAction] = buffNextReward
                else:
                    # get the next fixed Q value with action that generates the max Q value
                    nextQValSelected = np.max(nextQVal[expInd])
                    resultedQFunction[expInd, buffAction] = discount * nextQValSelected + buffNextReward
                expInd = expInd + 1


            sess.run(forward_step,feed_dict={ target: resultedQFunction, in_layer: observedVal})

        if done is True:
            totalReward = 0
            # calculate the discounted total reward
            for s in range(t+1):
                totalReward += discount**s
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
        sess.run(TargetNetworkbout.assign(bout))

print(rewardLst)
sess.close()
