# This script reads images from the data folders, prepossesses the data with cropping, resizing, noising and rotation.
# it than creates and returns batches of key-value-pairs from the processed images with their corresponding labels

# TODO crop, resize, noise, rotation, etc.
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

AUTOTUNE = tf.data.experimental.AUTOTUNE

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import config as cfg

# set directory of training images
train_data_dir = cfg.train_dir

# set directory of test images
test_data_dir = cfg.test_dir

# data directory for preparing data (either train or test data)
train_data_dir = pathlib.Path(train_data_dir)
test_data_dir = pathlib.Path(test_data_dir)

# get possible breed names
CLASS_NAMES = np.array([item.name for item in test_data_dir.glob('*')])
print('breed names: ', CLASS_NAMES)

# define parameters
BATCH_SIZE = cfg.batch_size
IMG_HEIGHT = cfg.img_height
IMG_WIDTH = cfg.img_width

# create dataset of file paths
train_list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*.jpg'))
test_list_ds = tf.data.Dataset.list_files(str(test_data_dir / '*/*.jpg'))
for f in train_list_ds.take(5):
    print('train image path: ', f.numpy())

for f in test_list_ds.take(5):
    print('test image path: ', f.numpy())


# convert file path to an (img, label) pair
def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] == CLASS_NAMES


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
for image, label in train_labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
test_labeled_ds = test_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
for image, label in test_labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # if cache:
    #     if isinstance(cache, str):
    #         ds = ds.cache(cache)
    #     else:
    #         ds = ds.cache()

    # allocate buffer
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # re-initialize the dataset
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # fetch batches in the background while the model is training
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_ds = prepare_for_training(train_labeled_ds)
test_ds = prepare_for_training(test_labeled_ds)

# set variables for plot (either train or test dataset)
image_batch, label_batch = next(iter(train_ds))
# image_batch, label_batch = next(iter(test_ds))


# define plots
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(15, 15))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis('off')

    return plt.show()


# show results
# show_batch(image_batch.numpy(), label_batch.numpy())

model_new = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Dropout(0.2),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model_new.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model_new.summary()

train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
test_image_count = len(list(test_data_dir.glob('*/*.jpg')))

logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

STEPS_PER_EPOCH = np.ceil(train_image_count / BATCH_SIZE)
val_steps = np.ceil(test_image_count / BATCH_SIZE)

model_new.fit(
    train_ds,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=1,
    callbacks=[tensorboard_callback],
    validation_steps= val_steps,
    validation_data=test_ds,
    validation_freq=10
)