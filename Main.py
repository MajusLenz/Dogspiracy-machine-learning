from all_breeds_array_for_random_agent import get_breeds
from RandomAgent import model

def main():
    breeds = get_breeds()
    random_breed = model(breeds)
    print(random_breed)

main()