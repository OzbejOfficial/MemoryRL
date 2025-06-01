# test_dqn_agent.py

import torch
import numpy as np
from environment import MemoryGameEnv
from dqn_agent import DQNAgent, DQN
from config import *
from utils import to_tensor  # make sure to extract to_tensor() into utils.py

def test_agent(num_games=100):
    env = MemoryGameEnv(num_pairs=DECK_PAIRS)
    total_cards = DECK_PAIRS * 2
    action_space_size = len(env.action_space())

    # Load agent and model
    agent = DQNAgent(state_size=total_cards * 2, action_space_size=action_space_size)
    agent.policy_net.load_state_dict(torch.load("memory_dqn.pt"))
    agent.policy_net.eval()

    total_steps = 0
    for _ in range(num_games):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            state_tensor = to_tensor(state, total_cards)
            valid_pairs = env.action_space()
            valid_idxs = [env.get_action_index(p) for p in valid_pairs]
            action_idx = agent.select_action(state_tensor, valid_idxs)
            action = env.get_action_from_index(action_idx)
            state, _, done, _ = env.step(*action)
            steps += 1
        total_steps += steps

    avg_steps = total_steps / num_games
    print(f"✅ Tested on {num_games} games")
    print(f"📊 Average steps per game: {avg_steps:.2f}")
    print(f"📉 Optimal is {DECK_PAIRS} matches = {DECK_PAIRS} steps (best case)")

if __name__ == "__main__":
    test_agent()
