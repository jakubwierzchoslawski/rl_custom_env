from os import replace
import random
import gymnasium as gym
import numpy as np
from model.Actor import Actor
from commons.visuals import Visuals

from model.Action import Action
from services.DecisionMaker import DecisionMaker


class EnvData:
    def __init__(
        self, width, height, exploration_rate, init_actors_nr, max_actors_per_cell
    ):
        self.BOARD_WIDTH = width
        self.BOARD_HEIGHT = height
        self.EXPLORATION_RATE = exploration_rate
        self.INIT_ACTORS_NR = init_actors_nr
        self.MAX_ACTORS_PER_CELL = max_actors_per_cell


class LifeExperienceEnv(gym.Env):

    def __init__(self, env_data):
        super(LifeExperienceEnv, self).__init__()
        self.ENV = env_data
        self.BOARD = np.full(
            (self.ENV.BOARD_WIDTH, self.ENV.BOARD_HEIGHT, self.ENV.MAX_ACTORS_PER_CELL),
            None,
            dtype=Actor,
        )

        # active actors on the board
        self.ACTORS = []
        # list of actors that were removed from the board during the decision=replace_actor or ...
        self.ACTORS_REMOVED = []

        # gymnasium observation space
        self.RL_observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(
                self.ENV.BOARD_WIDTH,
                self.ENV.BOARD_HEIGHT,
                self.ENV.MAX_ACTORS_PER_CELL,
            ),
            dtype=np.float32,
        )
        # gymnasium action space
        self.RL_action_space = gym.spaces.Discrete(
            4
        )  # 0: Up, 1: Down, 2: Left, 3: Right

        # initialize visuals
        self.vis = Visuals(self.ENV, self.BOARD, self.ACTORS)

        self.initialize_actors(self.ENV.INIT_ACTORS_NR)

        self.act = Action(self.ENV, self.BOARD)
        self.dm = DecisionMaker(self.ENV, self.BOARD)

        # Initialize Q-table
        num_states = self.define_number_of_states()
        num_actions = 4  # Assuming 4 actions: Up, Down, Left, Right
        self.q_table = np.zeros((num_states, num_actions))

        # Learning parameters
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def reset(self):
        self.BOARD = np.full(
            (self.ENV.BOARD_WIDTH, self.ENV.BOARD_HEIGHT, self.ENV.MAX_ACTORS_PER_CELL),
            None,
            dtype=Actor,
        )
        return self.BOARD

    def step(self):
        temp_actors = self.ACTORS.copy()

        while len(temp_actors) > 0:
            actor = temp_actors[0]

            state = self.get_state_for_cell(actor)

            # Choose an action for the actor
            action, _ = self.act.choose_action(actor, state)
            new_x, new_y = self.act.calculate_new_position(
                actor.position[0], actor.position[1], action
            )
            decision = self.dm.make_decision(new_x, new_y, actor)

            # Perform the action and update actor's state
            self.perform_action(actor, action, decision)

            # Get the new state and reward
            new_state = self.get_state_for_cell(actor.position[0], actor.position[1])
            reward = self.calculate_reward(actor, decision)

            # Q-learning update
            old_value = self.q_table[state, action]
            next_max = np.max(self.q_table[new_state])
            new_value = (1 - self.alpha) * old_value + self.alpha * (
                reward + self.gamma * next_max
            )
            self.q_table[state, action] = new_value

            temp_actors.remove(actor)
        # Return the updated state and reward

        # self.vis.print_board()
        return self.BOARD

    def map_color_to_int(self, color):
        # Example mapping, modify as per your color set
        color_map = {"RED": 1, "GREEN": 2}
        return color_map.get(color, 0)  # Default to 0 if color not found

    def perform_action(self, actor, action, decision):
        new_x, new_y = self.act.calculate_new_position(
            actor.position[0], actor.position[1], action
        )

        # Use DecisionMaker to get the decision type
        decision_type = self.dm.make_decision(new_x, new_y, actor)

        if decision_type == "STAY_AT_POSITION":
            self.calculate_reward(actor, decision)
            return

        elif decision_type == "TAKE_FREE_CELL_LAYER":
            free_layer = self.act.find_first_free_layer(new_x, new_y)
            self.move_actor_only(actor, new_x, new_y, free_layer)
            self.calculate_reward(actor, decision)

        elif decision_type == "REPLACE_EXISTING_ACTOR":
            layer = self.act.get_layer_for_removal(new_x, new_y, actor.color)
            self.move_and_replace_actor(actor, new_x, new_y, layer)
            self.calculate_reward(actor, decision)

    def calculate_reward(self, actor, decision):
        # Constants for rewards
        REWARD_STAY = 0
        REWARD_FREE_CELL = 1
        REWARD_REPLACE_ACTOR = 10
        REWARD_DEFAULT = -1
        # Map decisions to rewards
        reward_mapping = {
            self.dm.MOVE_DECISIONS[0]: REWARD_STAY,
            self.dm.MOVE_DECISIONS[1]: REWARD_FREE_CELL,
            self.dm.MOVE_DECISIONS[2]: REWARD_REPLACE_ACTOR,
        }

        # Get the reward for the given decision
        reward = reward_mapping.get(decision, REWARD_DEFAULT)

        # Update the actor's total rewards
        actor.total_rewards += reward

        return reward

    def initialize_actors(self, num_actors):
        available_positions = []
        for x in range(self.ENV.BOARD_WIDTH):
            for y in range(self.ENV.BOARD_HEIGHT):
                for layer in range(2):
                    if self.BOARD[x, y, layer] == None:
                        available_positions.append((x, y, layer))

        num_available_positions = len(available_positions)
        if num_available_positions == 0:
            # print("No available positions to add actors")
            return -1  # No available positions to add actors

        num_actors_to_add = min(num_actors, num_available_positions)
        # print("Number actors to add in initialization phase: ", num_actors_to_add)
        selected_positions = random.sample(available_positions, num_actors_to_add)

        for position in selected_positions:
            actor = Actor(position[:2])  # Actor position does not include layer
            x, y, layer = position
            actor.layer = layer
            self.BOARD[x, y, layer] = actor  # Occupy the chosen slot
            self.ACTORS.append(actor)
            print(
                f"Actor {actor.name} added to position {actor.position} at layer {actor.layer} "
            )
        # self.vis.render_3d_board_with_coloured_actors()
        print("Actors initialized \n\n\n")

    def move_actor_only(self, actor, new_x, new_y, new_layer):
        old_x, old_y, old_layer = actor.position[0], actor.position[1], actor.layer

        # Check if the new position is already occupied
        if self.BOARD[new_x, new_y, new_layer] is not None:
            print("Actor already exists at new position")
            return

        # Clear the old position
        self.BOARD[old_x, old_y, old_layer] = None

        # Update the actor's position and layer
        actor.position = [new_x, new_y]
        actor.layer = new_layer
        self.BOARD[new_x, new_y, new_layer] = actor

    def move_and_replace_actor(self, actor, new_x, new_y, new_layer):
        replaced_actor = self.BOARD[new_x, new_y, new_layer]

        if replaced_actor is not None:
            # Remove the replaced actor from the board and the ACTORS list
            self.ACTORS_REMOVED.append(replaced_actor)
            if replaced_actor in self.ACTORS:
                self.ACTORS.remove(replaced_actor)

            # Clear the replaced actor's position
            self.BOARD[
                replaced_actor.position[0],
                replaced_actor.position[1],
                replaced_actor.layer,
            ] = None

            # Clear the old position of the moving actor
            old_x, old_y, old_layer = actor.position[0], actor.position[1], actor.layer
            self.BOARD[old_x, old_y, old_layer] = None

            # Update the position and layer of the moving actor
            actor.position = [new_x, new_y]
            actor.layer = new_layer
            self.BOARD[new_x, new_y, new_layer] = actor

    def encode_actor(self, actor):
        num_color = 1 if actor is not None and actor.color == "RED" else 0
        # Encode the actor's attributes into a single number or a vector
        # For simplicity, let's assume it's just a combination of color and age

        # TODO pokombinowac z roznym kodowaniem i zobaczyc jak wplynie na wyniki uczenia
        return num_color + actor.age * actor.total_rewards

    def get_state_for_cell(self, actor):
        state = []
        free_layer = self.act.find_first_free_layer(actor.position[0], actor.position[1])
        
        if free_layer == -1:
            actor = self.BOARD[actor.position[0], actor.position[1], actor.layer]
            if actor is not None:
                state.append(1)  # Occupancy
            else:
                state.append(0)  # No occupancy
                state.append(0)  # Default value for empty cell

            print("###################################### state: ", state)
        return state
