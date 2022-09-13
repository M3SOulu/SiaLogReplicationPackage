from fileinput import filename
import numpy as np

from main import dump_dataset_info
import tqdm
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf

OUTPUT_DIRECTORY = 'deep_log'
MAX_SEQ_LEN = 19


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


def train_deep_log(epochs, dataset, output_directory, load_if_model_exists, k, file_name='deep_log.hdf5'):
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, Dropout, GRU, \
        Lambda
    train_x, train_y = dataset.deep_log_train(MAX_SEQ_LEN)
    val_x, val_y = dataset.deep_log_validation(MAX_SEQ_LEN)
    train_y -= 1
    val_y -= 1
    model = Sequential([
        Embedding(len(dataset.negative_events)+1, 16),
        # Lambda(lambda x: tf.expand_dims(x, -1), input_shape=(None,)),
        # Bidirectional(GRU(65, return_sequences=True)),
        LSTM(24, return_sequences=True),
        Dropout(0.5),
        LSTM(24),
        # Dropout(0.5),
        # Dense(128, activation=tf.nn.silu),
        # Dropout(0.5),
        # Dense(128, activation=tf.nn.silu),
        # Dropout(0.5),
        # Dense(128, activation=tf.nn.silu),
        Dropout(0.5),
        Dense(len(dataset.negative_events))
    ])
    model.compile(optimizer='rmsprop',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  )
    model.summary()
    if load_if_model_exists:
        try:
            model.load_weights(file_name)
            print("Weight successfully loaded")
            return model
        except Exception as e:
            print(f"Failed to load weights from {file_name}. Error: {e}")
            print("Training a new model...")
    model.fit(x=train_x, y=train_y, epochs=epochs, batch_size=16,
              # validation_split=0.5,
              validation_data=(val_x, val_y),
              callbacks=[
                  ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)
              ]
              )
    model.load_weights(file_name)
    return model


def evaluate_deep_log(deep_log_model, dataset, k, output_directory, verbose):
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    x_true, y_true = dataset.deep_log_test()
    y_pred = []

    tbl = {e: i for i, e in enumerate(dataset.negative_events, start=1)}
    for x, y in tqdm.tqdm(zip(x_true, y_true)):
        if any(e not in tbl for e in x):
            y_pred.append(1)
            continue
        x = [tbl[e] for e in x]
        sub_x, sub_y = list(zip(*[(s[:-1], s[-1]) for s in dataset.sub_sequences(x, MAX_SEQ_LEN)]))
        sub_y_hat = deep_log_model.predict(sub_x).argsort(-1)[:, -k:]
        faults = sum(0 if e in p else 1 for e, p in zip(sub_y, sub_y_hat + 1))
        y_hat = 1 if faults > 0 else 0
        y_pred.append(y_hat)

    print(precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def main():
    # imports
    from data import Dataset

    # hyper-parameters
    dataset_name = "hdfs"
    train_data_proportion = 0.90
    validation_data_proportion = 0.001
    use_validation_on_training = True
    evaluate_emb_model_after_training = True
    batch_size = 2048
    train_pair_generation_strategy = "sample"
    validation_pair_generation_strategy = "sample"
    test_pair_generation_strategy = "all"
    shuffler_buffer_size = 2 ** 33
    epochs = 72
    load_if_model_file_exists = False
    noisy_datasets_per_ratio = 5
    k = 4

    # preparing dataset
    dataset = Dataset(dataset_name=dataset_name,
                      train_proportion=train_data_proportion,
                      validation_proportion=validation_data_proportion,
                      batch_size=batch_size,
                      train_pair_generation_strategy=train_pair_generation_strategy,
                      validation_pair_generation_strategy=validation_pair_generation_strategy,
                      test_pair_generation_strategy=test_pair_generation_strategy,
                      shuffler_buffer_size=shuffler_buffer_size, remove_redundant=True)

    # print dataset info
    dump_dataset_info(dataset)

    # create output directory
    create_output_directory()

    deep_log_model = train_deep_log(epochs, dataset, OUTPUT_DIRECTORY, False, k, file_name='deep_log.hdf5')

    # evaluating classifiers
    print("Evaluating classifiers")
    evaluate_deep_log(deep_log_model, dataset,
                      k=k,
                      output_directory=OUTPUT_DIRECTORY,
                      verbose=1)


if __name__ == '__main__':
    main()
