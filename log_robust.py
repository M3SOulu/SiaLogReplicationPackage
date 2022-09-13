import numpy as np

from main import dump_dataset_info

OUTPUT_DIRECTORY = 'log_robust'


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


from tensorflow.keras.layers import *


def train_log_robust(epochs, dataset):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.losses import BinaryCrossentropy
    from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from metrics import F1Score

    # train_x = pad_sequences(np.concatenate(
    #     [dataset.train_data_table[0], dataset.train_data_table[1], dataset.validation_data_table[0],
    #      dataset.validation_data_table[1]]))
    # train_y = np.concatenate([np.zeros(len(dataset.train_data_table[0])), np.ones(len(dataset.train_data_table[1])),
    #                           np.zeros(len(dataset.validation_data_table[0])),
    #                           np.ones(len(dataset.validation_data_table[1]))])
    train_x, train_y = dataset.one_to_one({
        0: np.concatenate([dataset.train_data_table[0], dataset.validation_data_table[0]]),
        1: np.concatenate([dataset.train_data_table[1], dataset.validation_data_table[1]])
    })

    # train_x = pad_sequences(np.concatenate([dataset.train_data_table[0], dataset.train_data_table[1]]))
    # train_y = np.concatenate([np.zeros(len(dataset.train_data_table[0])), np.ones(len(dataset.train_data_table[1]))])

    # val_x = pad_sequences(np.concatenate([dataset.validation_data_table[0], dataset.validation_data_table[1]]))
    # val_y = np.concatenate([np.zeros(len(dataset.validation_data_table[0])), np.ones(len(dataset.validation_data_table[1]))])

    test_x = pad_sequences(np.concatenate([dataset.test_data_table[0], dataset.test_data_table[1]]))
    test_y = np.concatenate([np.zeros(len(dataset.test_data_table[0])), np.ones(len(dataset.test_data_table[1]))])

    print(f"{train_y.shape=} {test_y.shape=}")
    x = Input(shape=(None,))
    y = Embedding(dataset.max_event_number + 1, 16, mask_zero=True)(x)
    y = Bidirectional(LSTM(64, return_sequences=True))(y)
    a = Dropout(0.5)(y)
    a = Conv1D(1, 1, activation='tanh')(a)
    a = Lambda(lambda x: tf.tile(x, [1, 1, 128]))(a)
    y = Multiply()([y, a])
    y = GlobalAvgPool1D()(y)
    y = Dropout(0.5)(y)
    y = Dense(1, activation='sigmoid')(y)
    model = Model(inputs=[x], outputs=[y])

    # model = Sequential([
    #     Embedding(dataset.max_event_number + 1, 16, mask_zero=True),
    #     # Bidirectional(LSTM(128, return_sequences=True)),
    #     # LSTM(64, return_sequences=True),
    #     LSTM(64),
    #     # Dropout(0.5),
    #     # TimeDistributed(Dense(64, activation='tanh', use_bias=False)),
    #     # GlobalAvgPool1D(),
    #     # Dropout(0.5),
    #     Dense(64, tf.nn.silu),
    #     Dropout(0.5),
    #     Dense(64, tf.nn.silu),
    #     Dropout(0.5),
    #     Dense(1, activation=tf.nn.sigmoid)
    # ])
    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=[Precision(), Recall(), F1Score()])
    model.summary()
    # try:
    #     pass
    #     # model.load_weights(f'{OUTPUT_DIRECTORY}/model_checkpoint.h5')
    # except:
    model.fit(x=train_x, y=train_y,
              validation_data=(test_x, test_y),
              epochs=epochs,
              batch_size=1024,
              class_weight={0: 0.1, 1: 0.9},
              callbacks=[
                  ModelCheckpoint(monitor='val_f1_score', mode='max',
                                  filepath=f'{OUTPUT_DIRECTORY}/model_checkpoint.h5', save_best_only=True,
                                  save_weights_only=True,
                                  verbose=1)
              ])
    model.load_weights(f'{OUTPUT_DIRECTORY}/model_checkpoint.h5')
    model.evaluate(x=test_x, y=test_y)
    print("Generating noisy testsets")
    noisy_data_tables = {ratio: [dataset.create_noisy_data_table(ratio) for _ in range(5)] for
                         ratio in np.arange(0.05, 0.31, 0.05)}

    for ratio, datasets in noisy_data_tables.items():
        print("Testing ratio:", ratio)
        for noisy_data_table in datasets:
            d_x, d_y = dataset.one_to_one(noisy_data_table)
            model.evaluate(x=d_x, y=d_y)

    return model


def main():
    # imports
    from data import Dataset

    # hyper-parameters
    dataset_name = "hadoop"
    train_data_proportion = 0.87
    validation_data_proportion = 0.03
    use_validation_on_training = True
    evaluate_emb_model_after_training = True
    batch_size = 512
    train_pair_generation_strategy = "sample"
    validation_pair_generation_strategy = "sample"
    test_pair_generation_strategy = "all"
    shuffler_buffer_size = 2 ** 33
    epochs = 24
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

    train_log_robust(epochs, dataset)


if __name__ == '__main__':
    main()
