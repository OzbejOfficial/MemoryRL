# utils.py

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_tensor(state, total_cards, memory_map):
    """
    Returns a flat state tensor of shape [3 * total_cards]
    - [0:total_cards]        → matched or not (1 if visible, 0 if hidden)
    - [total_cards:2*...]    → memory values (card index +1), 0 = unknown
    - [2*total_cards:3*...]  → is-known memory mask
    """
    match_flags = np.array([1 if val != -1 else 0 for val in state])
    memory_values = np.zeros(total_cards)
    memory_known = np.zeros(total_cards)

    for i in range(total_cards):
        if i in memory_map:
            memory_values[i] = memory_map[i] + 1  # shift so 0 = unknown
            memory_known[i] = 1

    return torch.tensor(np.concatenate([match_flags, memory_values, memory_known]), dtype=torch.float32)

def stack_sequence(seq_list):
    """
    Given a list of [input_size] tensors, returns [1, T, input_size]
    """
    return torch.stack(seq_list).unsqueeze(0)
