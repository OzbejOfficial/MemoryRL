import numpy as np
import torch

def to_tensor(state, total_cards):
    onehot = np.zeros(total_cards * 2)
    for i, val in enumerate(state):
        if val == -1:
            onehot[i] = 1
        else:
            onehot[total_cards + i] = 1 + val
    return torch.tensor(onehot, dtype=torch.float32).unsqueeze(0)
