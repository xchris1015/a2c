"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
Mofan, pl for actor, td for critic
"""

import numpy as np
import tensorflow as tf
import gym
from World import World
import matplotlib.pyplot as plt

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 100
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 300  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

env = World(2, GAMMA)

num_agents = 2

N_F = num_agents
N_A = num_agents


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r * (1. - done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error
        self.n_step_error = tf.placeholder(tf.float32, None, "n_step_error")  # n_step_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=250,  # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss/ actor loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td_error):
        s = s[np.newaxis, :]  # make it as row vector by inserting an axis along first dimension (1, n_state)
        feed_dict = {self.s: s, self.a: a, self.td_error: td_error}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        # print(probs)
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, env, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.td_r = tf.placeholder(tf.float32, None, 'td_r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=250,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_td_error'):
            combine_rewards = 0
            for i, agent in enumerate(env.agents):
                combine_rewards += self.td_r[i] / agent.average_step_reward
            self.td_error = combine_rewards + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval = target - prediction
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, td_r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_r, self.train_op],
                                    {self.s: s, self.v_: v_, self.td_r: td_r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)


critic = Critic(env, sess, n_features=N_F, lr=LR_C)



# we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

best_selection_rates = []
for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    jump = 0
    track_r = []
    best_selections = []
    while True:

        #        times = 0
        #        five_step_rewards = []
        #        for i in range(num_agents):
        #            five_step_rewards.append([])
        #        five_step_dones = []

        #       while times < 5:
        a = actor.choose_action(s)
        s_, r, done, best_selection = env.step(a)
        # for i, agent_reward in r:
        #    five_step_rewards[i].append(agent_reward)
        # five_step_dones.append(done)
        # times+=1
        t += 1
        # five_step = discount_with_dones(five_step_rewards, five_step_dones, GAMMA)

        track_r.append(r[a])
        best_selections.append(best_selection)

        td_error = critic.learn(s, r, s_)

        actor.learn(s, a, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_

        best_selection_rate = float(sum(best_selections) / len(best_selections))
        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            print("episode:", i_episode, "  best_selection_rate:", best_selection_rate, " Jump-Times", jump)
            break

    best_selection_rates.append(best_selection_rate)

plt.plot(best_selection_rates)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Best_Selection_Rate')
plt.show()
