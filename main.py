from dogBreeds import getBreeds
from randomAgent import model

def main():
    breeds = getBreeds()
    randomBreed = model(breeds)
    print(randomBreed)

main()






