import numpy as np

from main import dump_dataset_info

OUTPUT_DIRECTORY = 'cnn_log'


def create_output_directory():
    # imports
    import os

    # define output directory name
    output_dir = OUTPUT_DIRECTORY

    # create directory if it doesn't exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # raise an exception if an entity exists with the same name but it's not a directory
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise FileExistsError(
            f"{output_dir} exists, but it is not a directory. Consider removing it or renaming the output directory")

    return output_dir


# noinspection DuplicatedCode
def train_cnn_log(epochs, dataset):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, Conv1D, Dense, GlobalAvgPool1D, GlobalMaxPool1D, GlobalMaxPooling1D, MaxPool1D, Dropout, Flatten
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.metrics import Precision, Recall
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from metrics import F1Score
    train_x = pad_sequences(np.concatenate([dataset.train_data_table[0], dataset.train_data_table[1], dataset.validation_data_table[0], dataset.validation_data_table[1]]))
    train_y = np.concatenate([np.zeros(len(dataset.train_data_table[0])), np.ones(len(dataset.train_data_table[1])), np.zeros(len(dataset.validation_data_table[0])), np.ones(len(dataset.validation_data_table[1]))])

    # val_x = pad_sequences(np.concatenate([dataset.validation_data_table[0], dataset.validation_data_table[1]]))
    # val_y = np.concatenate([np.zeros(len(dataset.validation_data_table[0])), np.ones(len(dataset.validation_data_table[1]))])

    test_x = pad_sequences(np.concatenate([dataset.test_data_table[0], dataset.test_data_table[1]]))
    test_y = np.concatenate([np.zeros(len(dataset.test_data_table[0])), np.ones(len(dataset.test_data_table[1]))])

    print(f"{train_y.shape=} {test_y.shape=}")

    model = Sequential([
        Embedding(dataset.max_event_number + 1, 16, mask_zero=True),

        Conv1D(16, 1, use_bias=False, activation=tf.nn.silu),
        # Dropout(0.5),

        # Conv1D(64, 3, use_bias=False, activation=tf.nn.silu),
        # Dropout(0.5),

        # Conv1D(64, 3, use_bias=False, activation=tf.nn.silu),
        # Dropout(0.5),
        #
        # Conv1D(64, 2, use_bias=False, activation=tf.nn.silu),
        # Dropout(0.5),

        GlobalMaxPool1D(),
        # GlobalAvgPool1D(),


        # Dense(64, tf.nn.silu),
        # Dropout(0.5),
        #
        # Dense(64, tf.nn.silu),
        # Dropout(0.5),

        Dense(16, tf.nn.silu),
        # Dropout(0.5),

        Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='rmsprop', loss=BinaryCrossentropy(), metrics=[Precision(), Recall(), F1Score()])
    model.summary()
    model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=epochs, batch_size=2048)
    # model.evaluate(x=test_x, y=test_y)
    return model


def main():
    # imports
    from data import Dataset

    # hyper-parameters
    dataset_name = "bgl"
    train_data_proportion = 0.87
    validation_data_proportion = 0.03
    use_validation_on_training = True
    evaluate_emb_model_after_training = True
    batch_size = 256
    train_pair_generation_strategy = "sample"
    validation_pair_generation_strategy = "sample"
    test_pair_generation_strategy = "all"
    shuffler_buffer_size = 2 ** 33
    epochs = 1024
    load_if_model_file_exists = False
    noisy_datasets_per_ratio = 5
    k = 3

    # preparing dataset
    dataset = Dataset(dataset_name=dataset_name,
                      train_proportion=train_data_proportion,
                      validation_proportion=validation_data_proportion,
                      batch_size=batch_size,
                      train_pair_generation_strategy=train_pair_generation_strategy,
                      validation_pair_generation_strategy=validation_pair_generation_strategy,
                      test_pair_generation_strategy=test_pair_generation_strategy,
                      shuffler_buffer_size=shuffler_buffer_size)

    # print dataset info
    dump_dataset_info(dataset)

    # create output directory
    create_output_directory()

    train_cnn_log(epochs, dataset)


if __name__ == '__main__':
    main()
