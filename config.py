# DATA PRE PROCESSING:

# directories where the data is located at
stanford_dir = "data/stanford/images/"
train_dir = "data/images/train/"
test_dir = "data/images/test/"

# this info comes from the dataset.
# To ensure that every breed has the same number of images to train and to test,
# this is set to the number of images of the breed with the fewest images.
max_number_of_images_per_breed = 148
max_number_of_train_images_per_breed = 118  # 80/20 Ratio

img_height = 224
img_width = 224

# LEARNING HYPER PARAMETERS:

batch_size = 16  # should be equal to the first param of the first model layer [ Conv2D(16, ...) ] (???)
learning_rate = 1e-4
