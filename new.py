import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from World import World

"""
Init Parameters
"""

# hyperparameters
hidden_size = 256 ## 200
learning_rate = 3e-4 ## 1e-3

# Constants
GAMMA = 0.99
num_steps = 300
max_episodes = 100 #1000
# CartPole-v0 defines “solving” as getting an average reward of 195.0 over 100 consecutive trials. Therefore a2c requires more episodes.


def encoder(position, size):
    result = np.zeros(size)
    result[position] = 1.0

    return result


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):

        """
        Init actor critic network with input

        :param num_inputs: Number of input
        :param num_actions: Two action in this case, left and right, which is 2
        :param hidden_size: 256
        :param learning_rate: 3e-4
        """

        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        """
        covert state to an torch Variable with float type, and put into actor and critic network. return Value and softmaxed policy distribution
        :param state: observation of the game.
                      Observation:
                    Type: Box(4)
                    Num	Observation                 Min         Max
                    0	Cart Position             -4.8            4.8
                    1	Cart Velocity             -Inf            Inf
                    2	Pole Angle                 -24 deg        24 deg
                    3	Pole Velocity At Tip      -Inf            Inf
        :return: value: single value
                 policy_dist: array with size of two
        """

        state = Variable(torch.from_numpy(state).float())
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist))

        return value, policy_dist


def a2c(env):

    """
    main function for a2c
    :param env:
            num_input = 4 (observations)
            num_output = 2 (actions)
    :return:
    """

    num_inputs = 2 ## number of observation
    num_outputs = 2




    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    ## placeholder

    all_reward = []
    entropy_term = 0

    for episode in range(max_episodes):

        rewards = []
        actions = []
        values = []
        log_probs = []

        state = env.reset()
        for steps in range(num_steps):
            value, policy_dist = actor_critic.forward(state)

        ## transfer tensor value and dist to single value and array
            value = value.detach().numpy()[0]
            dist = policy_dist.detach().numpy()

        ## selection action by its prob
            action = np.random.choice(num_inputs, p=np.squeeze(dist))
            ## take log of the action prob
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            ## calculate the entropy
            entropy = -np.sum(np.mean(dist) * np.log(dist))

            ## use the step function to get the next state and reward
            new_state, reward, done, _ = env.step(action)

            ## add reward, value and log_prob to buffer and update entropy. move to next state
            rewards.append(reward)
            actions.append(action)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state


            ## if done == true or this is last step:
            ## update reward for this eposide, steps that taken of this eposide
            if done or steps == num_steps - 1:
                Qval, _ = actor_critic.forward(new_state)
                Qval = Qval.detach().numpy()[0]
                ## take last 10 eposide length to calculate the average length

                break

        # compute Q values
        ## init target Q-value
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic, FloatTensor = tensor() with float type
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)

        ## calculate loss between target the actual
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

        final_reward = 0
        for agent in env.agents:
            final_reward += np.log(agent.step_reward / num_steps)

        sys.stdout.write("episode: {}, reward: {},  \n".format(episode, final_reward))






        ## zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).

        ## loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.

        ## opt.step() causes the optimizer to take a step based on the gradients of the parameters.

        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

    # Plot results
  #  smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
  #  smoothed_rewards = [elem for elem in smoothed_rewards]
 #   plt.plot(all_rewards)
 #   plt.plot(smoothed_rewards)
 #   plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()

if __name__ == "__main__":
    env = World(2,0.99)
    a2c(env)