# Sarvan Gill - MECH 590 Assignment 1
# October 30th, 2024

from enum import Enum
import random
import matplotlib.pyplot as plt
import numpy as np

# Create model of environment
blocks = [[0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,1,0,0,0], [0,0,1,1,0,1], [0,0,0,0,0,0], [0,0,0,1,0,0]]
num_blocks = sum([sum(i) for i in blocks])
gamma = 0.95

#start and end positions
start_x = 4
start_y = 0
goal_x = 2
goal_y = 4

# action probabilities
p_planned = 0.6
p_other = 0.3
p_nothing = 0.1

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

#rewards 
r_action = -1
r_nothing = -2
r_goal = 100

# Create structures for policy and value function and initialize them
policy_init = {}
v_init = {}
for y, row in enumerate(blocks):
    for x, val in enumerate(row):
        if val == 0:
            policy_init[f"{x},{y}"] = Action.LEFT
            v_init[f"{x},{y}"] = 0

# Roll a dice to see if the attempted action happens, if not determine what happens instead
def attempt_action(action):
    action_mod = 0
    do_nothing = False
    roll = random.randint(1,10)
    if roll > p_planned * 10:
        if roll > 10 - 10*p_nothing:
            do_nothing = True
        else:
            action_mod = roll - p_planned * 10

    action = Action((action.value + action_mod) % 4)

    return action

# Given an action and a state this gives the next state (this does not include the randomness)
def get_new_state(state_x, state_y, action):
    new_state_x = state_x
    new_state_y = state_y
    if action == Action.LEFT:
        new_state_x = state_x - 1
        if state_x == 0 or blocks[state_y][new_state_x] == 1:
            new_state_x = state_x
    elif action == Action.RIGHT:
        new_state_x = state_x + 1
        if state_x == 5 or blocks[state_y][new_state_x] == 1:
            new_state_x = state_x
    elif action == Action.DOWN:
        new_state_y = state_y + 1
        if state_y == 5 or blocks[new_state_y][state_x] == 1:
            new_state_y = state_y
    elif action == Action.UP:
        new_state_y = state_y - 1
        if state_y == 0 or blocks[new_state_y][state_x] == 1:
            new_state_y = state_y

    return (new_state_x, new_state_y)

def visualizePolicy(policy, title):
    arrows = {"R":(1,0), "L":(-1,0),"U":(0,1),"D":(0,-1)}
    scale = 0.25
    
    ar = {key:policy[key].name[0] for key in policy}

    fig, ax = plt.subplots(figsize=(6, 6))
    for key in ar:
            if int(key[0]) == goal_x and int(key[2]) == goal_y:
                circle = plt.Circle((int(key[0]), 6 - int(key[2])), 0.3, color='g')
                ax.add_patch(circle)
                continue
            plt.arrow(int(key[0]), 6 - int(key[2]), scale*arrows[ar[key]][0], scale*arrows[ar[key]][1], head_width=0.1)
    
    for i, row in enumerate(blocks):
        for j, val in enumerate(row):
            if val:
                square = plt.Rectangle((j - 0.5, 6 - i -0.5), 1, 1, color='r')
                ax.add_patch(square)
    
    plt.title(title)
    plt.show()



############################ Model Based Policy iteration ############################

# Evaluate a given policy pi by calculating the value function for that policy
def policy_evaluation(policy, value_function):
    curr_v_pi = value_function

    while True:
        new_v_pi = {}
        for state in policy:
            state_x = int(state[0])
            state_y = int(state[2])

            new_v_pi[f"{state_x},{state_y}"] = curr_v_pi[state]

            if state_x == goal_x and state_y == goal_y:
                continue
            action = policy[state]
            new_state_x, new_state_y = get_new_state(state_x, state_y, action)
            
            r = r_action if state_x+state_y != new_state_x+new_state_y else r_nothing
            if new_state_x == goal_x and new_state_y == goal_y:
                r = r_goal
            
            sum = 0
            for taken_action in Action:
                s_prime_x, s_prime_y = get_new_state(state_x, state_y, taken_action)
                if taken_action == action:
                    p = p_planned
                else:
                    p = p_nothing
                if s_prime_x == state_x and s_prime_y == state_y:
                    continue
                sum = sum + p * curr_v_pi[f"{s_prime_x},{s_prime_y}"]
            
            sum = gamma*sum
            
            new_v_pi[f"{state_x},{state_y}"] = r + sum

        if new_v_pi == curr_v_pi:
            break
        else:
            curr_v_pi = new_v_pi
    
    return curr_v_pi

v_pi = policy_evaluation(policy_init, v_init)


# Policy improvement
# First we build q_pi from v_pi
def q_pi_from_v_pi(v_pi):
    q_pi = {}
    for state in v_pi:
        for action in Action:
            state_x = int(state[0])
            state_y = int(state[2])
            sum = 0

            for taken_action in Action:
                s_prime_x, s_prime_y = get_new_state(state_x, state_y, taken_action)
                if taken_action == action:
                    p = p_planned
                else:
                    p = p_nothing

                r = r_action if state_x+state_y != s_prime_x+s_prime_y else r_nothing
                if s_prime_x == goal_x and s_prime_y == goal_y:
                    r = r_goal
                
                if r == r_nothing:
                    sum = sum + p*r
                else:
                    sum = sum + p * (r + gamma*v_pi[f"{s_prime_x},{s_prime_y}"])
            
            q_pi[f"{state_x},{state_y},{action}"] = sum

    return q_pi

def policy_improvement(pi_init, v_init):
    curr_pi = pi_init

    while True:
        next_pi = {}
        v_pi = policy_evaluation(curr_pi, v_init)
        q_pi = q_pi_from_v_pi(v_pi)

        for state in curr_pi:
            state_x = int(state[0])
            state_y = int(state[2])
            best_q = q_pi[f"{state_x},{state_y},{Action.LEFT}"]
            best_action = Action.LEFT

            next_pi[f"{state_x},{state_y}"] = best_action
            
            if state_x == goal_x and state_y == goal_y:
                continue
            
            for action in Action:
                q = q_pi[f"{state_x},{state_y},{action}"]

                if q > best_q:
                    best_q = q
                    best_action = action
            
            next_pi[f"{state_x},{state_y}"] = best_action
                
        if curr_pi == next_pi:
            break
        else:
            curr_pi = next_pi

    return curr_pi, v_pi

optimal_policy, v_pi = policy_improvement(policy_init, v_init)
visualizePolicy(optimal_policy, "Optimal Policy via Policy Improvement")



############################ Model Based Value iteration ############################

def get_optimal_value_function(v_init):
    curr_v = v_init
    while True:
        next_v = curr_v

        for state in curr_v:
            state_x = int(state[0])
            state_y = int(state[2])
            max_val = r_nothing
            for action in Action:
                sum = 0

                for taken_action in Action:
                    s_prime_x, s_prime_y = get_new_state(state_x, state_y, taken_action)
                    if taken_action == action:
                        p = p_planned
                    else:
                        p = p_nothing

                    r = r_action if state_x+state_y != s_prime_x+s_prime_y else r_nothing
                    if s_prime_x == goal_x and s_prime_y == goal_y:
                        r = r_goal
                    
                    if r == r_nothing:
                        sum = sum + p*r
                    else:
                        sum = sum + p * (r + gamma*v_pi[f"{s_prime_x},{s_prime_y}"])

                if sum > max_val:
                    max_val = sum
            
            next_v[state] = max_val
        
        if next_v == curr_v:
            break
        else:
            curr_v = next_v
    
    return curr_v

def get_pi_from_v(v_pi):
    q_pi = q_pi_from_v_pi(v_pi)
    pi = {}
    for state in v_pi:
            state_x = int(state[0])
            state_y = int(state[2])
            best_q = q_pi[f"{state_x},{state_y},{Action.LEFT}"]
            best_action = Action.LEFT

            pi[state] = best_action
            
            if state_x == goal_x and state_y == goal_y:
                continue
            
            for action in Action:
                q = q_pi[f"{state_x},{state_y},{action}"]

                if q > best_q:
                    best_q = q
                    best_action = action
            
            pi[f"{state_x},{state_y}"] = best_action

    return pi


v_star = get_optimal_value_function(v_init)
v_iter_optimal_pi = get_pi_from_v(v_star)

visualizePolicy(v_iter_optimal_pi, "Optimal policy via Value Iteration")

