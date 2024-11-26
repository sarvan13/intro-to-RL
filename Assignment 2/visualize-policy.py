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

def plot_training_error(training_error, title):
      error = [training_error[i] for i in range(int(len(training_error) / 10))]
      plt.plot(error)
      plt.title(title)
      plt.show()



json_file = 'dyna_q_test.json'
np_file = 'dyna_q_test.npy'

with open(json_file, 'rb') as f:
     policy_json = json.load(f)

context = {"np": np}

policy = {eval(k, context): Actions(v) for k, v in policy_json.items()}

rewards = np.load(np_file)

visualizePolicy(policy, "Dyna-Q policy")
plot_training_error(rewards, "Cumulative Reward")
