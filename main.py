from all_breeds_array_for_random_agent import getBreeds
from randomAgent import model

def main():
    breeds = getBreeds()
    randomBreed = model(breeds)
    print(randomBreed)

main()
