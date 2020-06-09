# This script reads images from the data folders, prepossesses the data with cropping, resizing, noising and rotation.
# it than creates and returns batches of key-value-pairs from the processed images with their corresponding labels

# TODO crop, resize, noise, rotation, etc.

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import config as cfg


# set directory of training images
train_dir = cfg.train_dir

data_dir = pathlib.Path(train_dir)

# get possible breed names
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
# print(CLASS_NAMES)

# define parameters
BATCH_SIZE = cfg.batch_size
IMG_HEIGHT = cfg.img_height
IMG_WIDTH = cfg.img_width

# create dataset of file paths
list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*.jpg'))

# for f in list_ds.take(5):
#     print(f.numpy())

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
labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
    # # This is a small dataset, only load it once, and keep it in memory.
    # # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # # fit in memory.
    # if cache:
    #     if isinstance(cache, str):
    #         ds = ds.cache(cache)
    #     else:
    #         ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


train_ds = prepare_for_training(labeled_ds)

image_batch, label_batch = next(iter(train_ds))


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(15, 15))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())
        plt.axis('off')

    return plt.show()

# show results
show_batch(image_batch.numpy(), label_batch.numpy())

