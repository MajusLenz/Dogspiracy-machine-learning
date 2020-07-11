from datetime import datetime

# directories where the data is located at
raw_dir = "data/flowers/"
train_dir = "data/images/train/"
test_dir = "data/images/test/"
validate_dir = "data/images/predict/"
predict_dir = "data/images/predict/"

# data directory where the model is saved to and loaded from
saved_model_dir = "saved_model/"

# this info comes from the dataset.
# To ensure that every breed has the same number of images to train and to test,
# this is set to the number of images of the breed with the fewest images.
max_number_of_images_per_breed = 633
max_number_of_train_images_per_breed = 506  # 80/20 Ratio


# LEARNING HYPER PARAMETERS:
img_height = 224
img_width = 224
batch_size = 64
number_of_epochs = 200
learning_rate = 0.001


# PARAMS TO CHOOSE CONTROL FLOW IN Main.py:

# Shall model be trained, validated or shall a prediction be made
# "train"       := train model and save it afterwards
# "evaluate"    := evaluate model with validation data
# "predict"     := predict classes of images in predict-folder
# "cli"         := action is set via cli-argument instead, when starting Main.py.   Example:  Main.py train
action = "cli"


model_name = "Adam_CategoricalCrossentropy_Conv7-16_Conv5-32_Conv3-64_Dense512_Dropout"
model_name += "_IH" + str(img_height) + "_IW" + str(img_width) + "_BSize" + str(batch_size) + "_LR" + str(learning_rate)
model_name += datetime.now().strftime("%Y%m%d-%H%M%S")

# Shall model be loaded or created?
# False         := create new model instead of loading one
# "MODELNAME"   := load model with this name instead of creating new one.  Example: model_name_to_be_loaded = "my_model"
model_name_to_be_loaded = 'MODEL_NAME_TO_BE_SAVED' # "Adam_CategoricalCrossentropy_Conv7-16_Conv5-32_Conv3-64_Dense512_Dropout_IH224_IW224_BSize32_LR0.00120200708-204419"

# name of the model that gets saved. Careful: Existing model with this name will be overwritten!
model_name_to_be_saved = model_name
