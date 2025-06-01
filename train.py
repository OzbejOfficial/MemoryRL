# train.py

import torch
import numpy as np
from environment import MemoryGameEnv
from dqn_agent import DQNAgent
from config import *
from utils import to_tensor, stack_sequence

def train():
    env = MemoryGameEnv(num_pairs=DECK_PAIRS)
    total_cards = DECK_PAIRS * 2
    input_dim = total_cards * 3
    output_dim = len(env.action_space())
    agent = DQNAgent(input_dim, output_dim)

    for ep in range(EPISODES):
        state = env.reset()
        memory = {}
        done = False
        h = None
        state_seq = []

        total_reward = 0
        while not done:
            state_tensor = to_tensor(state, total_cards, memory)
            state_seq.append(state_tensor)
            valid_pairs = env.action_space()
            valid_idxs = [env.get_action_index(p) for p in valid_pairs]

            seq_tensor = stack_sequence(state_seq)
            action_idx = agent.select_action(seq_tensor, valid_idxs, h)
            action = env.get_action_from_index(action_idx)
            next_state, reward, done, (i1, v1), (i2, v2) = env.step(*action)

            # update memory
            memory[i1] = v1
            memory[i2] = v2

            next_tensor = to_tensor(next_state, total_cards, memory)
            next_seq = state_seq + [next_tensor]

            agent.store((stack_sequence(state_seq), action_idx, reward, stack_sequence(next_seq), float(done)))

            state = next_state
            state_seq = next_seq
            total_reward += reward
            agent.train_step()

        agent.eps = max(EPS_END, agent.eps * EPS_DECAY)
        if ep % TARGET_UPDATE == 0:
            agent.update_target()
        if ep % 50 == 0:
            print(f"Ep {ep:>4} | Reward: {total_reward:.2f} | Eps: {agent.eps:.3f}")
            torch.save(agent.model.state_dict(), f"memory_dqn_lstm_ep{ep}.pt")
            print(f"📦 Saved checkpoint: memory_dqn_lstm_ep{ep}.pt")

    torch.save(agent.model.state_dict(), "memory_dqn_lstm.pt")
    print("✅ Saved model to memory_dqn_lstm.pt")

if __name__ == "__main__":
    train()
