# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from replay_buffer import ReplayBuffer
from config import *

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_space_size):
        self.policy_net = DQN(state_size, action_space_size)
        self.target_net = DQN(state_size, action_space_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.eps = EPS_START
        self.action_space_size = action_space_size
        self.update_target()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state_tensor, valid_action_idxs):
        if random.random() < self.eps:
            return random.choice(valid_action_idxs)
        with torch.no_grad():
            q_vals = self.policy_net(state_tensor)  # shape: [1, N]
            q_vals = q_vals.squeeze(0)              # shape: [N]
            mask = torch.ones(self.action_space_size, dtype=torch.bool)
            mask[valid_action_idxs] = False         # False = valid, True = invalid
            q_vals[mask] = -float('inf')
            return torch.argmax(q_vals).item()

    def store(self, *args):
        self.buffer.push(*args)

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Current Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Max Q-value for next state
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
        expected_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and optimization
        loss = nn.MSELoss()(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

