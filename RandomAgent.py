import random


class RandomAgent:

    def __init__(self, breeds):
        self.breeds = breeds

    def sample(self):
        random_breed_index = random.randint(0, len(self.breeds) -1)
        return self.breeds[random_breed_index]
