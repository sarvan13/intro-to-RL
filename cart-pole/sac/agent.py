import torch
import torch.optim as optim
import torch.nn as nn
from networks import ActorNet, QNet, ValueNet
import gymnasium as gym
from collections import deque, namedtuple
import random

class SACAgent():
    def __init__(self, max_action, state_dims, action_dims, alr, qlr, vlr, batch_size=256,
                 rewards_scale = 2, alpha = 0.2, gamma=0.99, tau=0.005, mem_length=1e5):
        self.actor = ActorNet(alr,state_dims, action_dims, max_action)
        self.q = QNet(qlr, state_dims, action_dims)
        self.value = ValueNet(vlr, state_dims)
        self.value_target = ValueNet(vlr, state_dims)
        self.value_target.load_state_dict(self.value.state_dict())

        self.max_action = max_action
        
        self.mem_length = mem_length
        self.replay_buffer = []
        # self.data_point = namedtuple()
        # self.replay_buffer = deque()

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.rewards_scale = rewards_scale

        self.loss = nn.MSELoss()

    def save(self):
        self.actor.save()
        self.value.save()
        self.value_target.save()
        self.q.save()
    def load(self):
        self.actor.load()
        self.value.load()
        self.value_target.load()
        self.q.load()

    def remember(self, data_point):
        self.replay_buffer.append(data_point)
        if len(self.replay_buffer) > self.mem_length:
            self.replay_buffer.pop(0)

    def choose_action(self, state, reparameterize=False):
        state = torch.tensor([state], dtype=torch.float).to(self.actor.device)
        action, _ = self.actor.sample(state, reparameterize)

        return action.cpu().detach().numpy()[0]
    
    def train(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = random.sample(self.replay_buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.actor.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.actor.device)

        # Train Value Network
        # error = V(s) - E(Q(s,a) - log(pi(a|s)))
        sampled_actions, log_probs = self.actor.sample(states, reparameterize=False)
        q_v = self.q.forward(states, sampled_actions)
        v = self.value.forward(states)
        next_v = q_v - log_probs
        v_loss = 0.5*self.loss(v, next_v.detach())
        self.value.optimizer.zero_grad()
        v_loss.backward()
        self.value.optimizer.step()

        #Train Actor Network
        # error = log(pi(a|s)) - q(s,a)
        sampled_actions, log_probs = self.actor.sample(states, reparameterize=True)
        q_actor = self.q.forward(states, sampled_actions)
        actor_loss = 0.5*self.loss(log_probs, q_actor)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Train Q Network
        # error = Q(s,a) - (r(s,a) + gamma* E(V'(s)))
        q = self.q.forward(states, actions)
        next_value = self.value_target.forward(next_states)
        next_value = (1 - dones).view(-1,1) * next_value
        q_target = self.rewards_scale*rewards.view(-1, 1) + self.gamma * next_value
        q_loss = 0.5*self.loss(q,q_target.detach())
        self.q.optimizer.zero_grad()
        q_loss.backward()
        self.q.optimizer.step()

        # Update V Target Network
        # Update the target value network

        for target_param, param in zip(self.value_target.parameters(), self.value.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



