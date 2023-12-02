import numpy as np
import string
import random
import uuid
import commons.nums as nums


class Actor:
    def __init__(self, position):
        self.actor_id = self.__generate_unique_id()
        self.color = np.random.choice(['RED', 'GREEN'])
        self.age = 0
        self.lifespan = nums.gausianlike_probability_distribution(0, 130, 70, 10)
        self.name = self.__generate_name(self.color)
                
        self.layer = 0
        self.position = position  # A tuple (x, y)
        self.historic_actions = []  # Historic actions buffer
        self.total_rewards = 0.0

    def __generate_unique_id(self):
        # Generate a unique ID using uuid library
        unique_id = str(uuid.uuid4())
        return unique_id

    def perform_action(self, action):
        # Perform the action

        # Append the action to the historic actions buffer
        self.historic_actions.append(action)

    def __generate_name(self, color, length=3):
        #characters = string.ascii_letters + string.digits
        characters = string.digits
        random_string = color + "_" + ''.join(random.choice(characters) for _ in range(length))
        
        return random_string

    def __update_reward(self, reward):
        self.total_rewards += reward
        
    def __str__(self) -> str:
        return "Actor: " + self.name + " | Color: " + self.color + " | Age: " + str(self.age) + " | Lifespan: " + str(self.lifespan) + " | Position: " + str(self.position) + " | Historic Actions: " + str(self.historic_actions) + " | Total Rewards: " + str(self.total_rewards)
        

