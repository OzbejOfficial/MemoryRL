# gui_game.py

import sys
import random
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QGridLayout, QLabel
from PyQt5.QtCore import QTimer
from environment import MemoryGameEnv
from dqn_agent import DQNAgent, DQN
from config import *
import torch
import numpy as np

CARD_SIZE = 80

def to_tensor(state, total_cards):
    onehot = np.zeros(total_cards * 2)
    for i, val in enumerate(state):
        if val == -1:
            onehot[i] = 1
        else:
            onehot[total_cards + i] = 1 + val
    return torch.tensor(onehot, dtype=torch.float32).unsqueeze(0)

class MemoryGameUI(QWidget):
    def __init__(self):
        super().__init__()
        self.env = MemoryGameEnv(num_pairs=DECK_PAIRS)
        self.total_cards = DECK_PAIRS * 2
        self.agent = DQNAgent(self.total_cards * 2, len(self.env.action_space()))
        self.agent.policy_net.load_state_dict(torch.load("memory_dqn.pt"))
        self.agent.policy_net.eval()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Memory Game — You vs AI")
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.status = QLabel("Your Turn")
        self.layout.addWidget(self.status, DECK_PAIRS, 0, 1, self.total_cards)

        self.buttons = []
        self.first_pick = None
        self.player_turn = True
        self.state = self.env.reset()

        for i in range(self.total_cards):
            btn = QPushButton("❓")
            btn.setFixedSize(CARD_SIZE, CARD_SIZE)
            btn.clicked.connect(lambda checked, i=i: self.card_clicked(i))
            self.layout.addWidget(btn, i // 6, i % 6)
            self.buttons.append(btn)

        self.update_board()

    def card_clicked(self, index):
        if not self.player_turn or self.env.matched[index]:
            return

        if self.first_pick is None:
            self.first_pick = index
            self.buttons[index].setText(str(self.env.cards[index]))
        else:
            second_pick = index
            self.buttons[second_pick].setText(str(self.env.cards[second_pick]))

            c1, c2 = self.env.cards[self.first_pick], self.env.cards[second_pick]
            if c1 == c2:
                self.env.matched[self.first_pick] = True
                self.env.matched[second_pick] = True
                self.status.setText("Matched! You play again.")
                self.first_pick = None
                self.update_board()
                if all(self.env.matched):
                    self.status.setText("🎉 Game Over! You win!")
                return
            else:
                self.player_turn = False
                self.status.setText("No match. AI turn...")
                QTimer.singleShot(1000, lambda: self.hide_cards([self.first_pick, second_pick]))
                QTimer.singleShot(1500, self.agent_turn)

    def hide_cards(self, indices):
        for i in indices:
            self.buttons[i].setText("❓")
        self.first_pick = None

    def agent_turn(self):
        state_tensor = to_tensor(self.env._get_observation(), self.total_cards)
        valid_pairs = self.env.action_space()
        valid_idxs = [self.env.get_action_index(p) for p in valid_pairs]
        action_idx = self.agent.select_action(state_tensor, valid_idxs)
        i, j = self.env.get_action_from_index(action_idx)

        self.buttons[i].setText(str(self.env.cards[i]))
        self.buttons[j].setText(str(self.env.cards[j]))

        QTimer.singleShot(1000, lambda: self.resolve_agent_turn(i, j))

    def resolve_agent_turn(self, i, j):
        c1, c2 = self.env.cards[i], self.env.cards[j]
        if c1 == c2:
            self.env.matched[i] = True
            self.env.matched[j] = True
            self.status.setText("AI matched! Playing again...")
            self.update_board()
            if all(self.env.matched):
                self.status.setText("💀 Game Over! AI wins!")
                return
            QTimer.singleShot(1000, self.agent_turn)
        else:
            self.buttons[i].setText("❓")
            self.buttons[j].setText("❓")
            self.status.setText("Your Turn")
            self.player_turn = True
            self.update_board()

    def update_board(self):
        for i, btn in enumerate(self.buttons):
            if self.env.matched[i]:
                btn.setText(str(self.env.cards[i]))
                btn.setEnabled(False)
            else:
                if btn.text() != "❓":
                    btn.setText("❓")
                btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MemoryGameUI()
    window.show()
    sys.exit(app.exec_())
