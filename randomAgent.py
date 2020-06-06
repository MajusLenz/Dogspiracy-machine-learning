import random

def model(breeds):
    breed = random.randint(0, len(breeds) -1)
    return breeds[breed]