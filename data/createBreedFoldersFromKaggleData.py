import csv
import os
from shutil import copyfile


csv_path = "kaggle/csv/"
csv_file_path_train = csv_path + "Dog_Breed_trainingdata.csv"
csv_file_path_test = csv_path + "Dog_Breed_testdata_sorted.csv"

new_train_dir = "images/train"
new_test_dir = "images/test"

kaggle_train_dir = "kaggle/images/Dog_Breed_Training_Images"
kaggle_test_dir = "kaggle/images/Dog_Breed_Test_Images"


def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def create_breed_folders_for_one_csv_file(csv_file_path, new_image_dir, old_image_dir):
    first_line = True

    with open(csv_file_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in csvreader:
            if first_line:
                first_line = False
                continue

            breed = row[1]

            # react to possible commas in breed names
            second_value = row[2]
            second_value_is_still_breed = breed.find('"', 0) == 0
            if second_value_is_still_breed:
                breed = breed + second_value
            breed = breed.replace('"', '')

            create_folder_if_not_exists(new_image_dir + "/" + breed)

            # copy image to new dir
            try:
                dog_id = row[0]
                file_name = dog_id + ".jpg"
                file_path = old_image_dir + "/" + file_name
                destination = new_image_dir + "/" + breed + "/" + file_name
                copyfile(file_path, destination)
            except FileNotFoundError:
                pass


# execute for training data
create_breed_folders_for_one_csv_file(csv_file_path_train, new_train_dir, kaggle_train_dir)

# execute for test data
#create_breed_folders_for_one_csv_file(csv_file_path_test, new_test_dir, kaggle_test_dir)
