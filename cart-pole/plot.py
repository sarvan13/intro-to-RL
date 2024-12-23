import matplotlib.pyplot as plt
import numpy as np

dqn_rewards = np.array(np.load("/home/sarvan/Classes/intro-to-RL/cart-pole/dqn-full-1.npy"))
sac_rewards = np.array(np.load("/home/sarvan/Classes/intro-to-RL/cart-pole/sac-full-1.npy"))

dqn_mean_rewards = [np.mean(dqn_rewards[np.max((0,i - 30)): i]) for i in range(len(dqn_rewards))]
sac_mean_rewards = [np.mean(sac_rewards[np.max((0,i - 30)): i]) for i in range(len(sac_rewards))]

# Plot DQN rewards
plt.plot(dqn_rewards, alpha=0.2, label="Reward")
plt.plot(dqn_mean_rewards, label="Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Control for Discrete Action Inverted Pendulum")
plt.legend()
plt.show()

# Plot SAC rewards
plt.plot(sac_rewards, alpha=0.2, label="Reward")
plt.plot(sac_mean_rewards, label="Average Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("SAC Control for Continuous Action Inverted Pendulum")
plt.legend()
plt.show()