# Sarvan Gill - MECH 590 Assignment 1
# October 30th, 2024

from enum import Enum
import random
import matplotlib.pyplot as plt
import numpy as np

class Action(Enum):
            LEFT = 0
            RIGHT = 1
            UP = 2
            DOWN = 3

class Model:
    def __init__(self):
        # Create model of environment
        self.blocks = [[0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,1,0,0,0], [0,0,1,1,0,1], [0,0,0,0,0,0], [0,0,0,1,0,0]]
        self.num_blocks = sum([sum(i) for i in self.blocks])
        self.gamma = 0.95

        #start and end positions
        self.start_x = 4
        self.start_y = 0
        self.goal_x = 2
        self.goal_y = 4

        # action probabilities
        self.p_planned = 0.6
        self.p_other = 0.3
        self.p_nothing = 0.1

        #rewards 
        self.r_action = -1
        self.r_nothing = -2
        self.r_goal = 100

        # Create structures for policy and value function and initialize them
        self.policy_init = {}
        self.v_init = {}
        for y, row in enumerate(self.blocks):
            for x, val in enumerate(row):
                if val == 0:
                    self.policy_init[f"{x},{y}"] = Action.LEFT
                    self.v_init[f"{x},{y}"] = 0

    # Roll a dice to see if the attempted action happens, if not determine what happens instead
    def attempt_action(self, action):
        action_mod = 0
        do_nothing = False
        roll = random.randint(1,10)
        if roll > self.p_planned * 10:
            if roll > 10 - 10*self.p_nothing:
                do_nothing = True
            else:
                action_mod = roll - self.p_planned * 10

        action = Action((action.value + action_mod) % 4)

        return action

    # Given an action and a state this gives the next state (this does not include the randomness)
    def get_new_state(self, state_x, state_y, action):
        new_state_x = state_x
        new_state_y = state_y
        if action == Action.LEFT:
            new_state_x = state_x - 1
            if state_x == 0 or self.blocks[state_y][new_state_x] == 1:
                new_state_x = state_x
        elif action == Action.RIGHT:
            new_state_x = state_x + 1
            if state_x == 5 or self.blocks[state_y][new_state_x] == 1:
                new_state_x = state_x
        elif action == Action.DOWN:
            new_state_y = state_y + 1
            if state_y == 5 or self.blocks[new_state_y][state_x] == 1:
                new_state_y = state_y
        elif action == Action.UP:
            new_state_y = state_y - 1
            if state_y == 0 or self.blocks[new_state_y][state_x] == 1:
                new_state_y = state_y

        return (new_state_x, new_state_y)

    def visualizePolicy(self, policy, title):
        arrows = {"R":(1,0), "L":(-1,0),"U":(0,1),"D":(0,-1)}
        scale = 0.25
        
        ar = {key:policy[key].name[0] for key in policy}

        fig, ax = plt.subplots(figsize=(6, 6))
        for key in ar:
                if int(key[0]) == self.goal_x and int(key[2]) == self.goal_y:
                    circle = plt.Circle((int(key[0]), 6 - int(key[2])), 0.3, color='g')
                    ax.add_patch(circle)
                    continue
                plt.arrow(int(key[0]), 6 - int(key[2]), scale*arrows[ar[key]][0], scale*arrows[ar[key]][1], head_width=0.1)
        
        for i, row in enumerate(self.blocks):
            for j, val in enumerate(row):
                if val:
                    square = plt.Rectangle((j - 0.5, 6 - i -0.5), 1, 1, color='r')
                    ax.add_patch(square)
        
        plt.title(title)
        plt.show()
