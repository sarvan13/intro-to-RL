import gymnasium as gym
import gymnasium_env
import dill
import numpy as np

env = gym.make("gymnasium_env/GridWorld-v0", render_mode="human")

with open('q_learning_model_5.pkl', 'rb') as f:
     model = dill.load(f)

q_values = model.get_q()

policy = {state:np.argmax(q_values[state]) for state in q_values}

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = policy[observation]
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
        print("failed")
env.close()




