from helper import load_csv

def getBreeds():
    dataset = load_csv('Dog_Breed_id_mapping.csv')
    breeds = []

    for x in dataset:
        breeds.append(x[0])

    return breeds


