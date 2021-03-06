from datetime import datetime

# directories where the data is located at
data_location = "data/"
dataset_name = "dog-breeds"
raw_dir = data_location + dataset_name
validate_dir = data_location + "images/validate"
predict_dir = data_location + "images/predict"

# data directory where the model is saved to and loaded from
saved_model_dir = "saved_model/"

# LEARNING HYPER PARAMETERS:
optimizer = "adam"
img_height = 224
img_width = 224
batch_size = 64
number_of_epochs = 120
learning_rate = 1e-4
validation_freq = 10


# PARAMS TO CHOOSE CONTROL FLOW IN Main.py:

# Shall model be trained, validated or shall a prediction be made
# "train"       := train model and save it afterwards
# "evaluate"    := evaluate model with validation data
# "predict"     := predict class of one image in predict_dir
# "cli"         := action is set via cli-argument instead, when starting Main.py.   Example:  Main.py train
action = "cli"

new_model_name = optimizer + "-" + dataset_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")

# Shall model be loaded or created?
# False         := create new model instead of loading one
# "MODELNAME"   := load model with this name instead of creating new one.  Example: model_name_to_be_loaded = "my_model"
model_name_to_be_loaded = "rsmprop-dog-breeds-20200712-214402" # "adam-dog-breeds-20200712-224040"

# name of the model that gets saved. Careful: Existing model with this name will be overwritten!
model_name_to_be_saved = new_model_name
