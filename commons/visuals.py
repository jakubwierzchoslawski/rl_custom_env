import io
from PIL import Image
import numpy as np
import imageio
import os
import re
from natsort import natsorted
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Visuals:
    def __init__(self, ENV, BOARD, ACTORS, media_dir="B:\\development\\python\\media"):
        self.ENV = ENV
        self.BOARD = BOARD
        self.ACTORS = ACTORS
        self.media_dir = media_dir

    def print_actor_move(
        self,
        actor_name,
        actor_color,
        current_x,
        current_y,
        new_x,
        new_y,
        actor_layer,
        new_layer,
        action_text,
        decision_text,
        dissalowed_decision_text,
        current_reward,
        new_reward,
    ):
        # Print actor details
        print(f"--------------------Actor: {actor_name}, color: {actor_color}")
        print(
            f"Current Position: ({current_x}, {current_y}), Layer: {actor_layer}. New Position: ({new_x}, {new_y}), Layer: {new_layer}"
        )
        print(
            f"Dissalowed actions: {dissalowed_decision_text}. Action taken: {action_text}, Decision: {decision_text}"
        )
        print(f"Current Reward: {current_reward}, New Reward: {new_reward}\n")

    def print_board(self):
        print("Board State:")

        # Create and print the header row
        header_row = " " * 12  # Adjust the spacing based on your needs
        for x in range(self.ENV.BOARD_WIDTH):
            header_row += f" Col {x:<5}"
        print(header_row)

        # Iterate over each cell and prepare row strings
        for y in range(self.ENV.BOARD_HEIGHT):
            row_str = f"Row {y:<5}"  # Adjust the spacing based on your needs
            for x in range(self.ENV.BOARD_WIDTH):
                cell_contents = self.BOARD[x, y]

                # Create a string to represent the actors in the cell
                actors_str = ", ".join(
                    actor.name if actor is not None else "Empty"
                    for actor in cell_contents
                )
                row_str += f" [{actors_str:<15}]"

            # Print the row string
            print(row_str)
        print("\n")

    def find_actor_position(
        actor, board, board_width, board_height, max_actors_per_cell
    ):
        for x in range(board_width):
            for y in range(board_height):
                for layer in range(max_actors_per_cell):
                    if board[x, y, layer] == actor:
                        return (x, y, layer)
        return None  # Actor not found

    def render_3d_board_with_coloured_actors(self):
        fig = plt.figure()
        # Clear the current figure
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")

        # Iterate over each cell and layer with colors based on Actor presence and color
        for x in range(self.ENV.BOARD_WIDTH):
            for y in range(self.ENV.BOARD_HEIGHT):
                for z in range(self.ENV.MAX_ACTORS_PER_CELL):
                    actor = self.BOARD[x, y, z]
                    if actor is not None:
                        # Use the color attribute of the Actor
                        color = actor.color
                        alpha = 0.8
                    else:
                        # Default color for empty cells (invisible)
                        color = "lightgray"
                        alpha = 0  # Make empty cells invisible

                    # Plot each cell as a cube with the assigned color
                    ax.bar3d(x, y, z, 1, 1, 1, color=color, alpha=alpha)

        # Adjust the view angle to have the x-y rectangle in front and z-axis behind
        ax.view_init(
            elev=20, azim=-130
        )  # Elevation and angle adjusted for the desired view

        # Set labels
        ax.set_xlabel("Left-Right (X-axis)")
        ax.set_ylabel("Up-Down (Y-axis)")
        ax.set_zlabel("Layers (Z-axis)")
        plt.title("3D Board Visualization with Actor Colors")

        # Adjusting the ticks to reflect the board dimensions
        ax.set_xticks(np.arange(0.5, self.ENV.BOARD_WIDTH + 0.5))
        ax.set_yticks(np.arange(0.5, self.ENV.BOARD_HEIGHT + 0.5))
        ax.set_zticks(np.arange(1, self.ENV.MAX_ACTORS_PER_CELL + 1))

        # Labeling the ticks to start from 1 instead of 0
        ax.set_xticklabels(np.arange(1, self.ENV.BOARD_WIDTH + 1))
        ax.set_yticklabels(np.arange(1, self.ENV.BOARD_HEIGHT + 1))
        ax.set_zticklabels(np.arange(1, self.ENV.MAX_ACTORS_PER_CELL + 1))

        # Setting the limits for the axes
        ax.set_xlim([0, self.ENV.BOARD_WIDTH])
        ax.set_ylim([0, self.ENV.BOARD_HEIGHT])
        ax.set_zlim([0, self.ENV.MAX_ACTORS_PER_CELL])

        # Draw and pause to update the plot
        plt.draw()
        plt.pause(0.001)

        # Capture the plot as an image in memory
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)  # Close the figure
        buf.seek(0)
        image = np.array(Image.open(buf))
        buf.close()

        return image

    def render_animation(self, images, output_filename="animation.mp4"):
        # Create the full output path
        full_output_path = os.path.join(self.media_dir, output_filename)

        print("rendering animation, full_output_path: ", full_output_path)

        # Create an animation from the list of in-memory images
        with imageio.get_writer(full_output_path, fps=10) as writer:
            for image in images:
                writer.append_data(image)

        print(f"Animation saved as {full_output_path}")

    def print_actor_total_reward(self, removed_actors, survived_actors):
        print("Removed Actors:")
        for actor in removed_actors:
            print(f"{actor.name}: {actor.total_rewards}")

        print("\n\nSurvived Actors:")
        for actor in survived_actors:
            print(f"{actor.name}: {actor.total_rewards}")
