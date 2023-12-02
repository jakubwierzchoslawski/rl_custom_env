class DecisionMaker:
    def __init__(self, env_data, board):
        self.ENV = env_data
        self.BOARD = board
        self.MOVE_DECISIONS = {
            0: "STAY_AT_POSITION",
            1: "TAKE_FREE_CELL_LAYER",
            2: "REPLACE_EXISTING_ACTOR",
        }

    def make_decision(self, new_x, new_y, actor):
        if self.is_out_of_bounds(new_x, new_y):
            return self.MOVE_DECISIONS[0]  # "STAY_AT_POSITION"

        if self.is_free_layer_available(new_x, new_y):
            return self.MOVE_DECISIONS[1]  # "TAKE_FREE_CELL_LAYER"

        if self.is_actor_of_same_color_in_cell(new_x, new_y, actor):
            return self.MOVE_DECISIONS[2]  # "REPLACE_EXISTING_ACTOR"

        return self.MOVE_DECISIONS[0]  # Default to "STAY_AT_POSITION"

    def is_out_of_bounds(self, x, y):
        return x < 0 or x >= self.ENV.BOARD_WIDTH or y < 0 or y >= self.ENV.BOARD_HEIGHT

    def is_free_layer_available(self, x, y):
        for layer in range(self.ENV.MAX_ACTORS_PER_CELL):
            if self.BOARD[x, y, layer] is None:
                return True
        return False

    def is_actor_of_same_color_in_cell(self, x, y, actor):
        for layer in range(self.ENV.MAX_ACTORS_PER_CELL):
            existing_actor = self.BOARD[x, y, layer]
            if existing_actor is not None and existing_actor.color == actor.color:
                return True
        return False
