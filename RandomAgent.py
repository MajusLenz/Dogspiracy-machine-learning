import random


class RandomAgent:

    def __init__(self, breeds):
        self.breeds = breeds

    # TODO define image type (numpy array??)
    def sample(self, image):
        random_breed_index = random.randint(0, len(self.breeds) -1)
        return self.breeds[random_breed_index]
