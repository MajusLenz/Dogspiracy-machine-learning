import tensorflow as tf
import pathlib
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE
import os
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from datetime import datetime


def main():
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

    # data_dir = pathlib.Path(
    #     "/media/mati3230/d8196a77-44d0-4208-a7d3-713df9c812f5/projects/test_keras/datasets/flower_photos")

    import config as cfg
    # data_dir = pathlib.Path(cfg.train_dir)

    data_dir = pathlib.Path(
        "data/flowers")
    image_count = len(list(data_dir.glob('*/*.jpg')))

    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    BATCH_SIZE = 64
    IMG_HEIGHT = 224
    IMG_WIDTH = 224

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*.jpg'))
    for f in list_ds.take(5):
        print(f.numpy())

    def get_label(file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        # bool_vec = parts[-2] == CLASS_NAMES
        result = tf.where(parts[-2] == CLASS_NAMES)
        return result
        #return parts[-2] == CLASS_NAMES

    def decode_img(img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # scale image between 0 and 1
        img = img / 255.0
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

    for image, label in labeled_ds.take(10):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
        # This is a small dataset, only load it once, and keep it in memory.
        # use `.cache(filename)` to cache preprocessing work for datasets that don't
        # fit in memory.
        '''
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        '''

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    # load all images
    all_ds = prepare_for_training(labeled_ds)
    test_percentage = 0.2
    # absolute nr of examples that are used for testing
    n_test_examples = math.floor(test_percentage * image_count)
    # how many batches should be considered in one epoch?
    STEPS_PER_EPOCH = np.ceil((image_count - n_test_examples)/BATCH_SIZE)
    # split in test and train dataset
    test_ds = all_ds.take(n_test_examples)
    train_ds = all_ds.skip(n_test_examples)

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
        Dense(5, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"
            ),
        metrics=["accuracy"])

    model.summary()

    import platform

    logdir_first_part = "./logs/scalars/"
    now_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # fix very nasty Windows OS bug, that TF runs in,
    # if those folders do not exist before TensorBoard tries to create them
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

    for image, label in train_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    model.fit(
        train_ds,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=100,
        callbacks=[tensorboard_callback],
        validation_data=test_ds,
        validation_freq=10
        )

    model.save('saved_model/' + 'MODEL_NAME_TO_BE_SAVED' + '.h5')

if __name__ == "__main__":
    main()
