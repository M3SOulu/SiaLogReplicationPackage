def store_hyperparameters(hparams):
    from utils import dump_dict
    print("HyperParameters")
    for k, v in hparams.items():
        print(k, ":", v)
    with open(f"out/HyperParameters.txt", "w") as file:
        file.write(dump_dict(hparams))


def dump_dataset_info(dataset):
    print("Positive sample count:", dataset.train_positive_sample_count)
    print("Negative sample count:", dataset.train_negative_sample_count)
    print("Train pair count:", dataset.train_pairs_count)
    print("Validation Pair count:", dataset.validation_pairs_count)
    print("Test Pair count:", dataset.test_pairs_count)
    print("Events count:", dataset.max_event_number)


def create_output_directory():
    # imports
    import os

    # define output directory name
    output_dir = "class_net"

    # create directory if it doesn't exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # raise an exception if an entity exists with the same name but it's not a directory
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise FileExistsError(
            f"{output_dir} exists, but it is not a directory. Consider removing it or renaming the output directory")


def train_neural_network(epochs, dataset, verbose=1):
    # imports
    from models import EmbeddingAndClassifier
    from time import time
    from data import Dataset
    from sklearn.metrics import f1_score, precision_score, recall_score
    from metrics import F1Score
    import tensorflow as tf
    import tempfile
    import os
    import numpy as np

    # creating best weights file path
    prediction_model_file = os.path.join("e2e", "model.h5")
    temp_weights_file = os.path.join(tempfile.gettempdir(), f"best_weights_{int(time() * 1000)}.h5")

    # creating the embedding and prediction model
    _, model = EmbeddingAndClassifier(dataset.max_event_number + 1)
    model.compile("adam", tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()])

    # data
    train_x, train_y = Dataset.one_to_one(dataset.train_data_table)
    validation_data = Dataset.one_to_one(dataset.validation_data_table)
    test_x, test_y = Dataset.one_to_one(dataset.test_data_table)

    # training
    model.fit(
        # train data
        x=train_x,
        y=train_y,

        # validation data
        validation_data=(test_x, test_y),

        # batch size
        batch_size=128,

        # weights
        class_weight={0: 0.1, 1: 0.9},

        # epochs
        epochs=epochs,

        # callbacks
        callbacks=[

            # saving best weights
            tf.keras.callbacks.ModelCheckpoint(temp_weights_file,
                                               monitor="val_f1_score",
                                               mode="max",
                                               save_weights_only=True,
                                               save_best_only=True,
                                               verbose=verbose),

            # reducing learning rate on loss plateau
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", verbose=verbose),
        ],

        # verbose type
        verbose=verbose)

    # loading the best neural network's weights
    print("Loading the best neural network's weights")
    model.load_weights(temp_weights_file)

    # evaluating model
    print("Evaluating The Neural Network:")
    model.evaluate(test_x, test_y)
    x, y = dataset.test_samples
    y_pred = np.round(model.predict(x, batch_size=128))
    print("Precision is", precision_score(y, y_pred))
    print("Recall is", recall_score(y, y_pred))
    print("F1 score is", f1_score(y, y_pred))
    model.save(prediction_model_file)


def main():
    # imports
    from data import Dataset

    # hyper-parameters
    dataset_name = "hadoop"
    train_data_proportion = 0.9
    validation_data_proportion = 0.005
    use_validation_on_training = False
    evaluate_emb_model_after_training = True
    batch_size = 256
    # batch_size = 128 * distribute_strategy.num_replicas_in_sync
    train_pair_generation_strategy = "sample"
    validation_pair_generation_strategy = "sample"
    test_pair_generation_strategy = "all"
    shuffler_buffer_size = 2 ** 33
    epochs = 92
    load_if_model_file_exists = True
    noisy_datasets_per_ratio = 5

    # print info
    store_hyperparameters({name: value for name, value in locals().items() if name[0] != "_"})

    # create output directory
    create_output_directory()

    # preparing dataset
    dataset = Dataset(dataset_name=dataset_name, train_proportion=train_data_proportion,
                      validation_proportion=validation_data_proportion,
                      batch_size=batch_size,
                      train_pair_generation_strategy=train_pair_generation_strategy,
                      validation_pair_generation_strategy=validation_pair_generation_strategy,
                      test_pair_generation_strategy=test_pair_generation_strategy,
                      shuffler_buffer_size=shuffler_buffer_size)

    # print dataset info
    dump_dataset_info(dataset)

    # training neural network
    train_neural_network(epochs, dataset)


if __name__ == '__main__':
    main()
