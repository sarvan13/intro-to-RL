import gymnasium as gym
import gymnasium_env
import numpy as np

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        print("failed")
env.close()




