from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import random
from enum import Enum

import gymnasium as gym
import gymnasium_env
from tqdm import tqdm
import json


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class DynaQAgent:
    def __init__(
        self,
        env,
        n: int,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.n = n
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.total_reward = []
        self.reward_sum = 0
        self.model = {}

    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        """Updates the model"""
        if str((obs,action)) not in self.model:
            self.model[str((obs,action))] = {str((next_obs, reward)): 1}
        else:
            s_a_map = self.model[str((obs,action))]
            if str((next_obs, reward)) in s_a_map:
                s_a_map[str((next_obs, reward))] = s_a_map[str((next_obs, reward))] + 1
            else:
                s_a_map[str((next_obs, reward))] = 1

        """Simulate n steps in the model and update Q"""
        for i in range(self.n):
            state_action = random.choice(list(self.model.keys()))
            sim_state, sim_action = self.extract_state_value(state_action)
            sim_state_reward = random.choices(list(self.model[state_action].keys()), list(self.model[state_action].values()), k=1)[0]
            sim_next_state, sim_reward = self.extract_state_value(sim_state_reward)

            next_q_value = (not terminated) * np.max(self.q_values[sim_next_state])
            sim_temporal_difference = (
                sim_reward + self.discount_factor * next_q_value - self.q_values[sim_state][sim_action]
            )

            self.q_values[sim_state][sim_action] = (
                self.q_values[sim_state][sim_action] + self.lr * sim_temporal_difference
            )

        self.reward_sum += reward
        self.total_reward.append(self.reward_sum)
    
    def extract_state_value(self,key):
        string_split = key.split(")), ")
        obs_string = string_split[0][1:] + "))"
        reward_string = string_split[1][:-1]

        context = {"np": np}
        obs = eval(obs_string, context)
        reward = eval(reward_string)

        return (obs,reward)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)



env = gym.make("gymnasium_env/GridWorld-v0")

learning_rate = 0.001
n_episodes = 1_000_000
start_epsilon = 0.25
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.25

agent = DynaQAgent(
    env=env,
    n=5,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    #agent.decay_epsilon()

q_values = agent.q_values

policy = {state:Actions(np.argmax(q_values[state])) for state in q_values}


with open('dyna_q_test.json', 'w') as f:
     json.dump({str(key):policy[key].value for key in policy}, f)

np.save("dyna_q_test.npy", np.array(agent.total_reward))