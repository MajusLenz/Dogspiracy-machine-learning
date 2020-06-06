from _csv import reader

RES_PATH = "./Dog_Breed_Recognition_Competition_Datasets/"

# load csv file
def load_csv(filename):
    dataset = list()
    with open(RES_PATH + filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

