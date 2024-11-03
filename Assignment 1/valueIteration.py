import model
from model import Action
from policyIteration import PolicyIteration

model = model.Model()
policy_iter = PolicyIteration()

def get_optimal_value_function(v_init):
    curr_v = v_init
    while True:
        next_v = {}

        for state in curr_v:
            state_x = int(state[0])
            state_y = int(state[2])
            max_val = model.r_nothing
            for action in Action:
                sum = 0

                for taken_action in Action:
                    s_prime_x, s_prime_y = model.get_new_state(state_x, state_y, taken_action)
                    if taken_action == action:
                        p = model.p_planned
                    else:
                        p = model.p_nothing

                    r = model.r_action if state_x+state_y != s_prime_x+s_prime_y else model.r_nothing
                    if s_prime_x == model.goal_x and s_prime_y == model.goal_y:
                        r = model.r_goal
                    
                    if r == model.r_nothing:
                        sum = sum + p*r
                    else:
                        sum = sum + p * (r + model.gamma*curr_v[f"{s_prime_x},{s_prime_y}"])

                if sum > max_val:
                    max_val = sum
            
            next_v[state] = max_val
        
        if next_v == curr_v:
            break
        else:
            curr_v = next_v
    
    return curr_v

def get_pi_from_v(v_pi):
    q_pi = policy_iter.q_pi_from_v_pi(v_pi)
    pi = {}
    for state in v_pi:
            state_x = int(state[0])
            state_y = int(state[2])
            best_q = q_pi[f"{state_x},{state_y},{Action.LEFT}"]
            best_action = Action.LEFT

            pi[state] = best_action
            
            if state_x == model.goal_x and state_y == model.goal_y:
                continue
            
            for action in Action:
                q = q_pi[f"{state_x},{state_y},{action}"]

                if q > best_q:
                    best_q = q
                    best_action = action
            
            pi[f"{state_x},{state_y}"] = best_action

    return pi


v_star = get_optimal_value_function(model.v_init)
v_iter_optimal_pi = get_pi_from_v(v_star)
model.visualizePolicy(v_iter_optimal_pi, "Optimal policy via Value Iteration")

