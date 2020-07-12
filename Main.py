import sys
import numpy as np
import os
import math
import platform
import pathlib
from datetime import datetime
import config as cfg

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
AUTOTUNE = tf.data.experimental.AUTOTUNE


def main():
    # get config parameters
    IMG_HEIGHT = cfg.img_height
    IMG_WIDTH = cfg.img_width
    BATCH_SIZE = cfg.batch_size
    NUMBER_OF_EPOCHS = cfg.number_of_epochs
    LEARNING_RATE = cfg.learning_rate
    VALIDATION_FREQ = cfg.validation_freq
    OPTIMIZER = cfg.optimizer
    RAW_DATA_DIR = pathlib.Path(cfg.raw_dir)
    VALIDATE_DATA_DIR = pathlib.Path(cfg.validate_dir)
    PREDICT_DATA_DIR = pathlib.Path(cfg.predict_dir)
    SAVED_MODEL_DIR = cfg.saved_model_dir
    MODEL_NAME_TO_BE_SAVED = cfg.model_name_to_be_saved
    MODEL_NAME_TO_BE_LOADED = cfg.model_name_to_be_loaded
    ACTION = cfg.action

    # get CLI arguments
    cli_argument = None
    if len(sys.argv) > 1:
        cli_argument = sys.argv[1]

    CLASS_NAMES = np.array([item.name for item in RAW_DATA_DIR.glob('*') if item.name != "LICENSE.txt"])
    NUMBER_OF_CLASSES = len(CLASS_NAMES)
    print("total classes: " + str(NUMBER_OF_CLASSES))
    print('all class names: ', CLASS_NAMES)

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

    def prepare_for_training(ds, shuffle_buffer_size=1000, shuffle=True):
        if shuffle:
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(BATCH_SIZE)

        # `prefetch` lets the dataset fetch batches in the background while the model
        # is training.
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    if MODEL_NAME_TO_BE_LOADED:
        print("loading model: " + MODEL_NAME_TO_BE_LOADED)
        model = tf.keras.models.load_model(SAVED_MODEL_DIR + MODEL_NAME_TO_BE_LOADED + ".h5")

    else:
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
            Dropout(0.2),
            Dense(NUMBER_OF_CLASSES, activation="softmax")
            ])

        if OPTIMIZER == "adam":
            optimizerInstance = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        elif OPTIMIZER == "rsmprop":
            optimizerInstance = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
        else:
            print("unknown optimizer! Either choose 'adam' or 'rsmprop' as optimizer in config.py")
            exit()

        model.compile(optimizer=optimizerInstance,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"])

    model.summary()

    if ACTION == "train" or (ACTION == "cli" and cli_argument == "train"):
        print("start training the model")

        data_dir = RAW_DATA_DIR
        image_count = len(list(data_dir.glob('*/*.jpg')))

        list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*.jpg'))

        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

        # print some image paths
        for f in list_ds.take(5):
            print(f.numpy())

        # print some image shapes and labels
        for image, label in labeled_ds.take(10):
            print("Image shape: ", image.numpy().shape)
            print("Label: ", label.numpy())

        # load all images
        all_ds = prepare_for_training(labeled_ds)
        test_percentage = 0.2
        # absolute nr of examples that are used for testing
        n_test_examples = math.floor(test_percentage * image_count)
        # how many batches should be considered in one epoch?
        STEPS_PER_EPOCH = np.ceil((image_count - n_test_examples) / BATCH_SIZE)
        # split in test and train dataset
        test_ds = all_ds.take(n_test_examples)
        train_ds = all_ds.skip(n_test_examples)

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

        # train
        model.fit(
            train_ds,
            steps_per_epoch=STEPS_PER_EPOCH,
            epochs=NUMBER_OF_EPOCHS,
            callbacks=[tensorboard_callback],
            validation_data=test_ds,
            validation_freq=VALIDATION_FREQ
            )

        model.save(SAVED_MODEL_DIR + MODEL_NAME_TO_BE_SAVED + '.h5')

    elif ACTION == "evaluate" or (ACTION == "cli" and cli_argument == "evaluate"):
        print("start evaluating the model")

        data_dir = VALIDATE_DATA_DIR
        image_count = len(list(data_dir.glob('*/*.jpg')))

        list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*.jpg'))

        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

        # load all images
        validate_ds = prepare_for_training(labeled_ds)

        # evaluate
        results = model.evaluate(validate_ds, steps=image_count)
        print("")
        print("model_accuracy: " + str(results[1]))
        print("loss: " + str(results[0]))

    elif ACTION == "predict" or (ACTION == "cli" and cli_argument == "predict"):
        print("start prediction of one image")

        data_dir = PREDICT_DATA_DIR
        image_count = len(list(data_dir.glob('*.jpg')))
        if image_count > 1:
            print("you can only have one image in the predict directory. please remove the rest.")

        list_ds = tf.data.Dataset.list_files(str(data_dir / '*.jpg'))

        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

        # load all images
        predict_ds = prepare_for_training(labeled_ds, shuffle=False)

        # predict
        results = model.predict(predict_ds, steps=image_count)
        result = results[0]

        max_prediction = max(result)
        max_prediction_index = result.tolist().index(max_prediction)
        predicted_class = CLASS_NAMES[max_prediction_index]
        print("Class '" + predicted_class + "' was predicted")

    else:
        # unknown action
        if ACTION == "cli":
            print(
                "please set the action to be executed in this script via the cli-argument when executing this script.")
            print("Example:  Main.py train")
            print("Possible actions: 'train', 'evaluate', 'predict'. For more Information see config.py")
        else:
            print("unknown action! please set action to 'train', 'evaluate' or 'predict' in config.py")


if __name__ == "__main__":
    main()
