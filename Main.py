from allBreedsArray import get_breeds
from RandomAgent import RandomAgent


def main():
    breeds = get_breeds()
    random_agent = RandomAgent(breeds)
    random_breed = random_agent.sample()
    print(random_breed)


if __name__ == "__main__":
    main()
