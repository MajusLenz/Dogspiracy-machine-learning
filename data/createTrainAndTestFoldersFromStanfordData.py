import os
import re
import traceback
from shutil import copyfile

# this info comes from the dataset.
# To ensure that every breed has the same number of images to train and to test,
# this is set to the number of images of the breed with the fewest images.
max_number_of_images_per_breed = 148
# can be changed as needed.
max_number_of_train_images_per_breed = 118  # 80/20 Ratio
max_number_of_test_images_per_breed = max_number_of_images_per_breed - max_number_of_train_images_per_breed

# data destinations
new_train_dir = "images/train/"
new_test_dir = "images/test/"
stanford_dir = "stanford/images/"


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

    create_folder_if_not_exists(new_train_dir + clean_breed_name)
    create_folder_if_not_exists(new_test_dir + clean_breed_name)

    # iterate over all images inside a breed folder
    for image in os.scandir(stanford_dir + breed_name):

        if image_counter > max_number_of_images_per_breed:
            break

        destination_folder = new_train_dir

        if image_counter > max_number_of_train_images_per_breed:
            destination_folder = new_test_dir

        # copy image to new destination
        try:
            file_name = image.name
            location_path = stanford_dir + breed_name + "/" + file_name
            destination = destination_folder + clean_breed_name + "/" + file_name

            copyfile(location_path, destination)
            image_counter += 1
        except:
            traceback.print_exc()
