# environment.py

import random

class MemoryGameEnv:
    def __init__(self, num_pairs=6):
        self.num_pairs = num_pairs
        self.deck = [i for i in range(num_pairs)] * 2
        self.reset()

    def reset(self):
        self.cards = random.sample(self.deck, len(self.deck))
        self.matched = [False] * len(self.cards)
        self.steps = 0
        return self._get_observation()

    def _get_observation(self):
        return [self.cards[i] if self.matched[i] else -1 for i in range(len(self.cards))]

    def step(self, action1, action2):
        self.steps += 1
        c1, c2 = self.cards[action1], self.cards[action2]
        match = c1 == c2
        if match:
            self.matched[action1] = True
            self.matched[action2] = True
        done = all(self.matched)
        reward = 1 if match else -0.1
        obs = self._get_observation()
        return obs, reward, done, (c1, c2)

    def valid_actions(self):
        return [i for i, m in enumerate(self.matched) if not m]

    def action_space(self):
        valid = self.valid_actions()
        return [(i, j) for i in valid for j in valid if i < j]

    def get_action_index(self, pair):
        all_actions = self.action_space()
        return all_actions.index(pair)

    def get_action_from_index(self, idx):
        return self.action_space()[idx]
