# This script creates a model to classify dog-breeds from images or loads an existing model from the disk.
# It then either trains the model via tensorflow, or evaluates the model's quality.
# The performed action can be changed via config.py. (See: "PARAMS TO CHOOSE CONTROL FLOW IN Main.py")
import sys
from datetime import datetime
import platform

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

# tf.random.set_seed(1337)
AUTOTUNE = tf.data.experimental.AUTOTUNE

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import config as cfg

# get config parameters
saved_model_dir = cfg.saved_model_dir
train_data_dir = cfg.train_dir
test_data_dir = cfg.test_dir
validate_data_dir = cfg.validate_dir
predict_data_dir = cfg.predict_dir
IMG_HEIGHT = cfg.img_height
IMG_WIDTH = cfg.img_width
BATCH_SIZE = cfg.batch_size
NUMBER_OF_EPOCHS = cfg.number_of_epochs
LEARNING_RATE = cfg.learning_rate
MODEL_NAME_TO_BE_LOADED = cfg.model_name_to_be_loaded
MODEL_NAME_TO_BE_SAVED = cfg.model_name_to_be_saved
ACTION = cfg.action

train_data_dir = pathlib.Path(train_data_dir)
test_data_dir = pathlib.Path(test_data_dir)
validate_data_dir = pathlib.Path(validate_data_dir)
predict_data_dir = pathlib.Path(predict_data_dir)

# get CLI arguments
cli_argument = None
if len(sys.argv) > 1:
    cli_argument = sys.argv[1]

# get possible breed names
CLASS_NAMES = np.array([item.name for item in test_data_dir.glob('*')])
print('all breed names: ', CLASS_NAMES)

NUMBER_OF_CLASSES = len(CLASS_NAMES)

# prepare GPUs
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(
            len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

print("count of usable GPUs: ")
print(len(gpus))


# convert file path to an (img, label) pair
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    # bool_vec = parts[-2] == CLASS_NAMES
    result = tf.where(parts[-2] == CLASS_NAMES)
    return result


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # scale image between 0 and 1
    img = img / 255.0
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def process_path(file_path):
    this_label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, this_label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)  # Flip horizontally
    # image = tf.image.resize_with_crop_or_pad(image, 272, 272)  # Add 48 pixels of padding
    # image = tf.image.random_crop(image, size=[IMG_HEIGHT, IMG_WIDTH, 3])  # Random crop back to 224 x 224
    # image = tf.image.random_jpeg_quality(image, 80, 100)
    # image = tf.image.random_saturation(image, 0.8, 1.2)
    # image = tf.image.random_contrast(image, 0.8, 1.2)
    # image = tf.image.random_brightness(image, 0.2)  # Random brightness
    return image, label


def prepare_dataset(ds, shuffle=True, shuffle_buffer_size=1000, is_augment=False):
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        # re-initialize the dataset
        ds = ds.repeat()

    # call prepare_dataset with param is_augment=True for Training-Dataset
    # if is_augment:
        # ds = ds.map(augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)

    # fetch batches in the background while the model is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
num_parallel_calls_param = AUTOTUNE

# fix nasty bug, that occurs in older Windows Version if you try to parallelly process multiple images
#if platform.system() == "Windows":
#    num_parallel_calls_param = None

if MODEL_NAME_TO_BE_LOADED:
    print("loading model " + MODEL_NAME_TO_BE_LOADED)
    model = tf.keras.models.load_model(saved_model_dir + MODEL_NAME_TO_BE_LOADED + ".h5")

else:
    print("creating new model")
    '''
    model = Sequential([
        Conv2D(filters=32, kernel_size=7, padding='same', activation='relu',
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
               kernel_initializer="he_normal"),  # add "input_shape" because this is the first layer of the Net
        MaxPooling2D(padding="same"),
        Dropout(0.2),
        Conv2D(64, 5, padding="same", activation="relu", kernel_initializer="he_normal"),
        MaxPooling2D(padding="same"),
        Dropout(0.2),
        Conv2D(64, 3, padding="same", activation="relu", kernel_initializer="he_normal"),
        MaxPooling2D(padding="same"),
        Dropout(0.2),
        Flatten(),  # transform 2D to 1D
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(NUMBER_OF_CLASSES, activation='softmax')
    ])
    '''

    model = Sequential([
        Conv2D(16, 5, padding="same", activation="relu",
               input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(),
        Conv2D(32, 5, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(32, 5, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 5, padding="same", activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 5, padding="same", activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation="relu"),
        Dense(NUMBER_OF_CLASSES, activation="softmax")
    ])

    # CategoricalCrossentropy: Computes the crossentropy loss between the labels and predictions.
    # Use this crossentropy loss function when there are two or more label classes.
    # We expect labels to be provided in a one_hot representation.
    # categorical_accuracy: Calculates how often predictions matches one-hot labels.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"
                  ),
                  metrics=["accuracy"])
    '''
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0,
            reduction="auto",
            name="categorical_crossentropy",
        ),
        metrics=['categorical_accuracy', 'accuracy'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['categorical_accuracy', 'accuracy'])
     '''


model.summary()

if ACTION == "train" or (ACTION == "cli" and cli_argument == "train"):
    print("start training the model")

    # create datasets from file paths
    train_list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*.jpg'))
    test_list_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*.jpg'))
    train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=num_parallel_calls_param)
    test_labeled_ds = test_list_ds.map(process_path, num_parallel_calls=num_parallel_calls_param)

    # prepare dataset
    train_ds = prepare_dataset(train_labeled_ds, is_augment=True)
    test_ds = prepare_dataset(test_labeled_ds)

    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    test_image_count = len(list(test_data_dir.glob('*/*.jpg')))

    logdir_first_part = "./logs/scalars/"
    now_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # fix very nasty Windows 8 bug, that TF runs in, if those folders do not exist before TensorBoard tries to create them
    if platform.system() == "Windows":
        logdir_first_part = "logs/scalars/"
        time_folder_path = logdir_first_part + now_time
        deeper_folder_path = time_folder_path + "/train"
        deeper_folder_path2 = deeper_folder_path + "/plugins"
        deeper_folder_path3 = deeper_folder_path2 + "/profile"

        def create_folder_if_not_exists(folder_name):
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        create_folder_if_not_exists(time_folder_path)
        create_folder_if_not_exists(deeper_folder_path)
        create_folder_if_not_exists(deeper_folder_path2)
        create_folder_if_not_exists(deeper_folder_path3)

    logdir = logdir_first_part + now_time
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    STEPS_PER_EPOCH = np.ceil(train_image_count / BATCH_SIZE)
    val_steps = np.ceil(test_image_count / BATCH_SIZE)

    # image plotting in tensorboard
    # Sets up a timestamped log directory.
    # logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    # file_writer = tf.summary.create_file_writer(logdir)

    # with file_writer.as_default():
        # Don't forget to reshape.
        # dataset_array = list(train_ds.as_numpy_iterator())
        # images = np.reshape(dataset_array[0:25], (-1, 28, 28, 1))
        # tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

    # show label array of random image from the train dataset
    for image, label in train_labeled_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=NUMBER_OF_EPOCHS,
        callbacks=[tensorboard_callback],
        validation_steps=val_steps,
        validation_data=test_ds,
        validation_freq=5,
        # shuffle=True
    )

    model.save(saved_model_dir + MODEL_NAME_TO_BE_SAVED + '.h5')

elif ACTION == "evaluate" or (ACTION == "cli" and cli_argument == "evaluate"):
    print("start evaluating the model")

    # create validation dataset from file paths
    validate_list_ds = tf.data.Dataset.list_files(str(validate_data_dir / '*/*.jpg'))
    validate_labeled_ds = validate_list_ds.map(process_path, num_parallel_calls=num_parallel_calls_param)

    # prepare dataset
    validate_ds = prepare_dataset(validate_labeled_ds)

    results = model.evaluate(validate_ds, steps=500)
    print("")
    print("model_accuracy: " + str(results[1]))
    print("loss: " + str(results[0]))

elif ACTION == "predict" or (ACTION == "cli" and cli_argument == "predict"):
    print("start prediction")

    # create predict dataset from file paths
    predict_list_ds = tf.data.Dataset.list_files(str(predict_data_dir / '*.jpg'))
    predict_ds = predict_list_ds.map(process_path, num_parallel_calls=num_parallel_calls_param)

    # prepare dataset
    predict_ds = prepare_dataset(predict_ds, shuffle=False)

    _, _, files = next(os.walk(str(predict_data_dir)))
    number_of_images = len(files)

    results = model.predict(predict_ds, steps=number_of_images)

    for result_index, result in enumerate(results):
        max_prediction = max(result)
        max_prediction_index = result.tolist().index(max_prediction)
        predicted_class = CLASS_NAMES[max_prediction_index]

        image_name = files[result_index]
        print("Class '" + predicted_class + "' was predicted for image " + image_name)

else:
    # unknown action
    if ACTION == "cli":
        print("please set the action to be executed in this script via the cli-argument when executing this script.")
        print("Example:  Main.py train")
        print("Possible actions: 'train', 'evaluate', 'predict'. For more Information see config.py")
    else:
        print("unknown action! please set action to 'train', 'evaluate' or 'predict' in config.py")
