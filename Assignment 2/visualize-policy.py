import matplotlib.pyplot as plt
import dill
import numpy as np
import json
from enum import Enum
import ast

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

def visualizePolicy(policy, title):
        arrows = {"r":(1,0), "l":(-1,0),"u":(0,-1),"d":(0,1)}
        scale = 0.25
        
        ar = {key:policy[key].name[0] for key in policy}

        fig, ax = plt.subplots(figsize=(12, 12))
        for key in ar:
                if int(key[0]) == 11 and int(key[1]) == 0:
                    circle = plt.Circle((int(key[0]), 12 - int(key[1])), 0.3, color='g')
                    ax.add_patch(circle)
                    continue
                plt.arrow(int(key[0]), 12 - int(key[1]), scale*arrows[ar[key]][0], scale*arrows[ar[key]][1], head_width=0.1)
        
        # for i, row in enumerate(self.blocks):
        #     for j, val in enumerate(row):
        #         if val:
        #             square = plt.Rectangle((j - 0.5, 12 - i -0.5), 1, 1, color='r')
        #             ax.add_patch(square)
        
        plt.title(title)
        plt.show()

def plot_training_error(training_error, label, title):
      error = [training_error[i] for i in range(int(len(training_error)))]
      plt.plot(error, label=label)
      plt.axvline(x=500_000,color = 'b', linestyle='--')
      plt.axvline(x=1_500_000,color = 'b', linestyle='--')
      plt.axvline(x=3_000_000,color = 'b',  linestyle='--')
      plt.xlabel("Number of Steps")
      plt.ylabel("Cumulative Reward")
      plt.title(title)



json_file = 'q_learning_step_55.json'

with open(json_file, 'rb') as f:
     policy_json = json.load(f)

context = {"np": np}

policy = {eval(k, context): Actions(v) for k, v in policy_json.items()}



visualizePolicy(policy, "Q Learning Policy Epsilon = 0.25, Alpha = 0.001")
# rewards = np.load('dyna_q_decay_50_a01.npy')
# plot_training_error(rewards[:8_000_000], "Decaying Epsilon, alpha = 0.01", "Dyna Q in Grid World")

# rewards2 = np.load('dyna_q_const_50_a01.npy')
# plt.plot(rewards2[:8_000_000], label="Random Start Epsilon =  0.25, alpha = 0.01")

# rewards4 = np.load('dyna_q_const02_50_a01.npy')
# plt.plot(rewards4[:8_000_000], label="Random Start Epsilon =  0.20, alpha = 0.01")

# rewards5 = np.load('dyna_q_const03_50_a01.npy')
# plt.plot(rewards5[:8_000_000], label="Epsilon =  0.30, alpha = 0.01")

# rewards3 = np.load('dyna_q_step_const_start.npy')
# plt.plot(rewards3, label="Epsilon =  0.25, alpha = 0.001")

# rewardsw = np.load('dyna_q_const_start_50_a01.npy')
# plt.plot(rewardsw, label="Epsilon = 0.25, alpha = 0.01")

# plt.legend(loc="upper left")
# plt.show()
# 


# q_learning_reward = np.load('q_learning_rewards_7.npy')
# plot_training_error(q_learning_reward, "Q Learning epsilon decay, alpha=0.001", "Q Learning vs Dyna Q")

# plt.plot(rewards, label="Dyna Q Decaying Epsilon, alpha = 0.01")
# plt.plot(rewards3, label="Dyna Q Epsilon =  0.25, alpha = 0.001")

# q_learning_reward2 = np.load('q_learning_step_55.npy')
# plt.plot(q_learning_reward2, label="Q Learning Epsilon = 0.25, alpha=0.001")

# plt.legend(loc="upper left")
# plt.show()

