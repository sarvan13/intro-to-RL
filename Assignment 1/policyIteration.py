import model as md
from model import Action

class PolicyIteration:
    def __init__(self):
        self.model = md.Model()

    # Evaluate a given policy pi by calculating the value function for that policy
    def policy_evaluation(self, policy, value_function):
        curr_v_pi = value_function

        while True:
            new_v_pi = {}
            for state in policy:
                state_x = int(state[0])
                state_y = int(state[2])

                new_v_pi[f"{state_x},{state_y}"] = curr_v_pi[state]

                if state_x == self.model.goal_x and state_y == self.model.goal_y:
                    continue
                action = policy[state]
                new_state_x, new_state_y = self.model.get_new_state(state_x, state_y, action)
                
                r = self.model.r_action if state_x+state_y != new_state_x+new_state_y else self.model.r_nothing
                if new_state_x == self.model.goal_x and new_state_y == self.model.goal_y:
                    r = self.model.r_goal
                
                sum = 0
                for taken_action in Action:
                    s_prime_x, s_prime_y = self.model.get_new_state(state_x, state_y, taken_action)
                    if taken_action == action:
                        p = self.model.p_planned
                    else:
                        p = self.model.p_nothing
                    if s_prime_x == state_x and s_prime_y == state_y:
                        continue
                    sum = sum + p * curr_v_pi[f"{s_prime_x},{s_prime_y}"]
                
                sum = self.model.gamma*sum
                
                new_v_pi[f"{state_x},{state_y}"] = r + sum

            if new_v_pi == curr_v_pi:
                break
            else:
                curr_v_pi = new_v_pi
        
        return curr_v_pi


    # Policy improvement
    # First we build q_pi from v_pi
    def q_pi_from_v_pi(self, v_pi):
        q_pi = {}
        for state in v_pi:
            for action in Action:
                state_x = int(state[0])
                state_y = int(state[2])
                sum = 0

                for taken_action in Action:
                    s_prime_x, s_prime_y = self.model.get_new_state(state_x, state_y, taken_action)
                    if taken_action == action:
                        p = self.model.p_planned
                    else:
                        p = self.model.p_nothing

                    r = self.model.r_action if state_x+state_y != s_prime_x+s_prime_y else self.model.r_nothing
                    if s_prime_x == self.model.goal_x and s_prime_y == self.model.goal_y:
                        r = self.model.r_goal
                    
                    if r == self.model.r_nothing:
                        sum = sum + p*r
                    else:
                        sum = sum + p * (r + self.model.gamma*v_pi[f"{s_prime_x},{s_prime_y}"])
                
                q_pi[f"{state_x},{state_y},{action}"] = sum

        return q_pi

    def policy_improvement(self,pi_init, v_init):
        curr_pi = pi_init

        while True:
            next_pi = {}
            v_pi = self.policy_evaluation(curr_pi, v_init)
            q_pi = self.q_pi_from_v_pi(v_pi)

            for state in curr_pi:
                state_x = int(state[0])
                state_y = int(state[2])
                best_q = q_pi[f"{state_x},{state_y},{Action.LEFT}"]
                best_action = Action.LEFT

                next_pi[f"{state_x},{state_y}"] = best_action
                
                if state_x == self.model.goal_x and state_y == self.model.goal_y:
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


policy_iter = PolicyIteration()
optimal_policy, v_pi = policy_iter.policy_improvement(policy_iter.model.policy_init, policy_iter.model.v_init)
policy_iter.model.visualizePolicy(optimal_policy, "Optimal Policy via Policy Improvement")
