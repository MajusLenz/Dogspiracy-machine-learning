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

# change vars to sort the test-images too:
csv_file_path = csv_file_path_train
new_image_dir = new_train_dir
old_image_dir = kaggle_train_dir


def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


with open(csv_file_path, newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csvreader:
        breed = row[1]

        # react to possible commas in breed names
        second_value = row[2]
        second_value_is_still_breed = breed.find('"', 0) == 0
        if second_value_is_still_breed:
            breed = breed + second_value
        breed = breed.replace('"', '')

        create_folder_if_not_exists(new_image_dir + "/" + breed)


        if row[0] == "10004890":
            # copy image to new dir
            dog_id = row[0]
            file_name = dog_id + ".jpg"
            file_path = old_image_dir +  "/" + file_name
            destination = new_image_dir + "/" + breed + "/" + file_name
            copyfile(file_path, destination)

        # print(breed)
