# train.py

import torch
import numpy as np
from environment import MemoryGameEnv
from dqn_agent import DQNAgent
from config import *
from utils import to_tensor

def train():
    env = MemoryGameEnv(num_pairs=DECK_PAIRS)
    total_cards = DECK_PAIRS * 2
    action_space = env.action_space()
    agent = DQNAgent(state_size=total_cards * 2, action_space_size=len(action_space))

    for ep in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            state_tensor = to_tensor(state, total_cards)
            valid_pairs = env.action_space()
            valid_idxs = [env.get_action_index(p) for p in valid_pairs]
            action_idx = agent.select_action(state_tensor, valid_idxs)
            action = env.get_action_from_index(action_idx)
            next_state, reward, done, _ = env.step(*action)
            next_tensor = to_tensor(next_state, total_cards)
            agent.store(state_tensor.squeeze(0).numpy(), action_idx, reward, next_tensor.squeeze(0).numpy(), float(done))
            state = next_state
            agent.train_step()

        agent.eps = max(EPS_END, agent.eps * EPS_DECAY)
        if ep % TARGET_UPDATE == 0:
            agent.update_target()
        if ep % 100 == 0:
            print(f"Episode {ep}, eps={agent.eps:.2f}")

    # ✅ Save the model after training finishes
    torch.save(agent.policy_net.state_dict(), "memory_dqn.pt")
    print("✅ DQN model saved as memory_dqn.pt")

if __name__ == "__main__":
    train()
