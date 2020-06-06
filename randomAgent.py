import random

def model(breeds):
    breed = random.randint(1, len(breeds) -1)
    return breeds[breed]