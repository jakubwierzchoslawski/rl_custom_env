import torch
import random


class QTable:
    def __init__(self, num_states, num_actions):
        self.Q_TABLE = torch.zeros(num_states, num_actions)

    def update(self, state, action, reward, next_state, alpha, gamma):
        max_next_q = torch.max(self.Q_TABLE[next_state])
        current_q = self.Q_TABLE[state, action]
        new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_next_q)
        self.Q_TABLE[state, action] = new_q

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return torch.argmax(self.Q_TABLE[state]).item()
