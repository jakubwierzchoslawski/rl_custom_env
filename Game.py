import random
from model.Environment import LifeExperienceEnv, EnvData
from commons.visuals import Visuals

import matplotlib.pyplot as plt


# setup environment here
def config():
    # Initialize board params

    board_data = {"width": 8, "height": 8}

    # initialize actors params
    actors_data = {"num_actors": 100, "max_actors_per_cell": 2}

    # initialize RL params
    exporation_rate = 0.1
    rl_data = {"exploration_rate": exporation_rate}

    return EnvData(
        board_data["width"],
        board_data["height"],
        rl_data["exploration_rate"],
        actors_data["num_actors"],
        actors_data["max_actors_per_cell"],
    )


class Game:
    def __init__(self):
        self.config = config()

    def play_scenario_1(self):
        training_steps = 3
        game_env = LifeExperienceEnv(self.config)

        vis = Visuals(game_env.ENV, game_env.BOARD, game_env.ACTORS)

        # List to store images
        images = []

        for i in range(training_steps):
            print(f"--> Step {i + 1} \n")
            game_env.step()

            image = vis.render_3d_board_with_coloured_actors()
            images.append(image)

        # Generate animation
        vis.render_animation(images, f"animation_{random.randint(0, 1000)}.mp4")
        images.clear()

        vis.print_actor_total_reward(game_env.ACTORS_REMOVED, game_env.ACTORS)


# --------------------- Main execution --------------------- #
if __name__ == "__main__":
    game = Game()
    game.play_scenario_1()
