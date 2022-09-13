from fileinput import filename
import numpy as np

from main import dump_dataset_info
import tqdm
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import tensorflow as tf

OUTPUT_DIRECTORY = 'log_anomaly'
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


def train_log_anomaly(epochs, dataset, output_directory, load_if_model_exists, k, file_name='log_anomaly.hdf5'):
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, Dropout, GRU, \
        Lambda, Input, Concatenate
    from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
    from keras.callbacks import ModelCheckpoint
    train_x, train_y = dataset.deep_log_train(MAX_SEQ_LEN)
    train_y -= 1
    x = Input((None,))
    x_c = Lambda(lambda x: tf.one_hot(tf.cast(x - 1, tf.int64), len(dataset.negative_events)))(x)
    x_c = Lambda(lambda x: tf.reduce_sum(x, 1))(x_c)
    y = Embedding(len(dataset.negative_events)+1, 16)(x)
    y = LSTM(24, return_sequences=True)(y)
    # y = Dropout(0.5)(y)
    y = LSTM(24)(y)
    y = Concatenate()([y, x_c])
    y = Dropout(0.5)(y)
    y = Dense(32, activation='relu')(y)
    # y = Dense(32)(y)
    y = Dropout(0.5)(y)
    y = Dense(len(dataset.negative_events))(y)

    model = Model(inputs=[x], outputs=[y])
    model.compile(optimizer='rmsprop',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[SparseTopKCategoricalAccuracy(k)],
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
              # validation_split=0.2,
              callbacks=[
                  ModelCheckpoint(file_name, 'loss', save_best_only=True,
                                  save_weights_only=True, verbose=1)
              ])
    print("Loading best training weights")
    model.load_weights(file_name)
    return model


def evaluate_log_anomaly(log_anomaly_model, dataset, k, output_directory, verbose):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    x_true, y_true = dataset.deep_log_test()
    y_pred = []
    pos_faults = []
    neg_faults = []

    tbl = {e: i for i, e in enumerate(dataset.negative_events, start=1)}
    for x, y in tqdm.tqdm(zip(x_true, y_true)):
        if any(e not in tbl for e in x):
            y_pred.append(1)
            continue
        x = [tbl[e] for e in x]
        sub_x, sub_y = list(zip(*[(s[:-1], s[-1]) for s in dataset.sub_sequences(x, MAX_SEQ_LEN)]))
        sub_y_hat = log_anomaly_model.predict(sub_x).argsort(-1)[:, -k:]
        faults = sum(0 if e in p else 1 for e, p in zip(sub_y, sub_y_hat + 1))
        y_hat = 1 if faults > 0 else 0
        y_pred.append(y_hat)
        if y == 1:
            pos_faults.append(faults)
        else:
            neg_faults.append(faults)

    print(
        f"{np.mean(pos_faults)=}{np.min(pos_faults)=}{np.max(pos_faults)=} ,, {np.mean(neg_faults)=}{np.min(neg_faults)=}{np.max(neg_faults)=}")
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
    epochs = 82
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

    log_anomaly_model = train_log_anomaly(epochs, dataset, OUTPUT_DIRECTORY, load_if_model_exists=False, k=k, file_name='log_anomaly.hdf5')

    # evaluating classifiers
    print("Evaluating classifiers")
    evaluate_log_anomaly(log_anomaly_model, dataset,
                         k=k,
                         output_directory=OUTPUT_DIRECTORY,
                         verbose=1)


if __name__ == '__main__':
    main()
