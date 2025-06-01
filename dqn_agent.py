# dqn_agent.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import numpy as np
from replay_buffer import ReplayBuffer
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RecurrentDQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        out, h = self.lstm(x, h)  # x: [batch, seq_len, input_size]
        out = self.fc(out[:, -1, :])  # last timestep
        return out, h

class DQNAgent:
    def __init__(self, input_size, output_size):
        self.model = RecurrentDQN(input_size, output_size).to(device)
        self.target_model = RecurrentDQN(input_size, output_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.buffer = ReplayBuffer(BUFFER_SIZE)
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.eps = EPS_START
        self.output_size = output_size

    def select_action(self, state_seq, valid_idxs, h=None):
        state_seq = state_seq.to(device)
        self.model.eval()
        with torch.no_grad():
            q_vals, _ = self.model(state_seq, h)
            q_vals = q_vals.squeeze(0)
            mask = torch.ones(self.output_size, dtype=torch.bool)
            mask[valid_idxs] = False
            q_vals[mask] = -float('inf')

        self.model.train()

        if torch.rand(1).item() < self.eps:
            return np.random.choice(valid_idxs)
        return torch.argmax(q_vals).item()

    def store(self, trajectory):
        self.buffer.push(*trajectory)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(self):
        self.model.train()

        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)

        state_seq, actions, rewards, next_seq, dones = zip(*batch)

        # Remove extra dim: [1, T, F] → [T, F]
        state_seq = [s.squeeze(0) for s in state_seq]
        next_seq = [s.squeeze(0) for s in next_seq]

        # move sequences to GPU
        state_seq = pad_sequence([s.to(device) for s in state_seq], batch_first=True)
        next_seq = pad_sequence([s.to(device) for s in next_seq], batch_first=True)

        # move rest of the tensors
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        q_vals, _ = self.model(state_seq)
        q_vals = q_vals.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q, _ = self.target_model(next_seq)
            max_next_q = next_q.max(1)[0]
        expected = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_vals, expected)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
