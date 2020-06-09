# Run me one time after you moved the stanford image folders in the directory that is set in "stanford_dir",
# to create the train and test folder.

import os
import re
import traceback
from shutil import copyfile
import config as cfg

# see config.py
max_number_of_images_per_breed = cfg.max_number_of_images_per_breed
max_number_of_train_images_per_breed = cfg.max_number_of_train_images_per_breed
max_number_of_test_images_per_breed = max_number_of_images_per_breed - max_number_of_train_images_per_breed
train_dir = cfg.train_dir
test_dir = cfg.test_dir
stanford_dir = cfg.stanford_dir


def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


# delete_every_string_character_before_delimiter("test", "e") = "st"
def delete_every_string_character_before_delimiter(string, delimiter="-"):
    return re.sub(r'^.*?' + delimiter, '', string)


# iterate over all breed folders in the dataset
for stanford_breed_folder in os.scandir(stanford_dir):
    image_counter = 0
    breed_name = stanford_breed_folder.name
    clean_breed_name = delete_every_string_character_before_delimiter(breed_name)

    create_folder_if_not_exists(train_dir + clean_breed_name)
    create_folder_if_not_exists(test_dir + clean_breed_name)

    # iterate over all images inside a breed folder
    for image in os.scandir(stanford_dir + breed_name):

        if image_counter >= max_number_of_images_per_breed:
            break

        destination_folder = train_dir

        if image_counter >= max_number_of_train_images_per_breed:
            destination_folder = test_dir

        # copy image to new destination
        try:
            file_name = image.name
            location_path = stanford_dir + breed_name + "/" + file_name
            destination = destination_folder + clean_breed_name + "/" + file_name

            copyfile(location_path, destination)
            image_counter = image_counter +1
        except:
            traceback.print_exc()