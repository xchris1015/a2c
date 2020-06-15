import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class World(gym.Env):
    def __init__(self, num_agents, gamma):
        ## place to set parameters, missing some of the parameters right now
        self.gamma = gamma
        self.num_agents = num_agents
        ## put agents to an array
        self.agents = [Agent() for i in range(num_agents)]
        ## init agent state with encoder function, state = np.array([0,1]) or np.array([1,0])
        for i, agent in enumerate(self.agents):
            agent.state = np.random.choice(2)
            agent.step_reward = 0
        ## required variables for gym inheritance

        ## current state set to None, state == action == observation in this case, also use encoder function,
        # example np.array([0,1]), means there are two agents and the second agent been selected
        self.state = None
        self.time = 0
        ## reward map from paper, all the reward are fixed
        self.reward_map = dict({0: [1.50, 0.768],
                                1: [2.25, 1.00],
                                2: [1.25, 0.384],
                                3: [1.50, 1.12],
                                4: [1.75, 0.384],
                                5: [1.25, 1.12]})
    """
    reset function to reset all of the agent state
    """
    def reset(self):
        current_state = []
        for agent in self.agents:
            agent.state = np.random.choice(2)
            agent.step_reward = 0
            current_state.append(agent.state)

        self.state = current_state

        return np.array(current_state)

    def discount_with_dones(rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma * r * (1. - done)  # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]

    """
    use state transaction function to get new state for each agent, then test if it is done, get the reward for each step
    """
    def step(self, action):
        done = bool(self.time >= 300)

        if not done:
            reward = self.get_reward(action)
        else:
            reward = 0

        best_selection = True
        for i, agent in enumerate(self.agents):
            agent.state = self.get_new_state(agent)
            if reward < self.get_reward(i):
                best_selection = False
            if action == i:
                agent.step_reward += reward
            #print("Agent_",i,"    State: ","Good" if agent.state == 0 else "Bad", "   Agent_Reward:", self.get_reward(i))
        #print("Current Selection:", action)

        return np.array(self.state), reward, done, best_selection

    """
    get the new state for input agent
    """

    def get_new_state(self, agent):
        current_state = agent.state
        state_transition_model = np.random.choice(2, p=[0.9, 0.1])
        if state_transition_model == 1:
            if current_state == 1:
                new_state = 0
            else:
                new_state = 1
        else:
            new_state = current_state

        return new_state

    """
    get reward by reward map init on world
    """

    def get_reward(self, action):
        reward = 0
        for i, agent in enumerate(self.agents):
            if action == i:
                reward = self.reward_map[action][agent.state]

        return reward

    """
    encoder for state and action
    """

class Agent(object):
    def __init__(self):
        self.state = None
        self.step_reward = 0
