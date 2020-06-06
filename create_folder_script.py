import csv

with open('data/csv/datasets_309436_628824_Dog_Breed_Recognition_Competition_Datasets_Dog_Breed_id_mapping.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        # print(', '.join(row))
        name = row[0]
        second_value = row[1]
        is_still_name = not second_value.isnumeric()
        if is_still_name:
            name = name + second_value

        name = name.replace('"', '')

        print('"' + name + '",')
