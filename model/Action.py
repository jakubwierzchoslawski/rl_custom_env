import numpy as np


class Action:
    def __init__(self, ENV, BOARD):
        self.ENV = ENV
        self.BOARD = BOARD

    def calculate_new_position(self, x, y, action):
        directions = {
            0: (x, max(y - 1, 0)),  # Up
            1: (x, min(y + 1, self.ENV.BOARD_HEIGHT - 1)),  # Down
            2: (max(x - 1, 0), y),  # Left
            3: (min(x + 1, self.ENV.BOARD_WIDTH - 1), y),  # Right
        }
        return directions.get(
            action, (x, y)
        )  # Default to current position if invalid action

    def choose_action(self, actor, state):
        current_x, current_y = actor.position
        disallowed_actions = self.get_disallowed_actions(current_x, current_y)

        # Exploration vs Exploitation decision
        if np.random.rand() < self.ENV.EXPLORATION_RATE:  # Exploration
            action = np.random.choice(
                [a for a in range(4) if a not in disallowed_actions]
            )
        else:
            action = np.argmax(self.q_table[state])

        return action, disallowed_actions

    def get_disallowed_actions(self, x, y):
        disallowed_actions = []
        if x == 0:
            disallowed_actions.append(2)  # Left disallowed
        if x == self.ENV.BOARD_WIDTH - 1:
            disallowed_actions.append(3)  # Right disallowed
        if y == 0:
            disallowed_actions.append(0)  # Up disallowed
        if y == self.ENV.BOARD_HEIGHT - 1:
            disallowed_actions.append(1)  # Down disallowed
        return disallowed_actions

    def get_layer_for_removal(self, x, y, color):
        same_color_actors = [
            actor for actor in self.BOARD[x, y] if actor and actor.color == color
        ]

        if len(same_color_actors) == 1:
            return 0  # Only one actor of the same color

        # Return the layer of the older actor (TODO: Adjust as per requirement)
        return 0 if same_color_actors[0].age > same_color_actors[1].age else 1

    def find_first_free_layer(self, x, y):
        if self.is_out_of_bounds(x, y):
            raise ValueError("Position out of bounds")

        for layer in range(self.ENV.MAX_ACTORS_PER_CELL):
            if self.BOARD[x, y, layer] is None:
                return layer  # First free layer found

        return -1  # No free layer available

    def is_out_of_bounds(self, x, y):
        return x < 0 or x >= self.ENV.BOARD_WIDTH or y < 0 or y >= self.ENV.BOARD_HEIGHT
