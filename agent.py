# agent.py

import random
import itertools
from collections import defaultdict

class QAgent:
    def __init__(self, alpha=0.1, gamma=0.95, epsilon=0.2, memory_limit=4):
        self.q = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = {}  # index -> revealed value
        self.memory_limit = memory_limit

    def update_memory(self, index, value):
        if len(self.memory) >= self.memory_limit:
            self.memory.pop(next(iter(self.memory)))
        self.memory[index] = value

    def choose_action(self, state, valid):
        pairs = list(itertools.combinations(valid, 2))
        if random.random() < self.epsilon:
            return random.choice(pairs)

        qvals = [self.q[(state, a)] for a in pairs]
        return pairs[qvals.index(max(qvals))]

    def learn(self, state, action, reward, next_state, done):
        key = (state, action)
        next_qs = [self.q[(next_state, a)] for a in itertools.combinations(range(len(state)), 2) if a[0] != a[1]]
        max_q_next = max(next_qs, default=0)
        target = reward + self.gamma * max_q_next * (not done)
        self.q[key] += self.alpha * (target - self.q[key])

    def clear_memory(self):
        self.memory.clear()
