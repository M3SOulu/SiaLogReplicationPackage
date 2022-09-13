def create_output_directory():
    # imports
    import os

    # define output directory name
    output_dir = "e2e"

    # create directory if it doesn't exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # raise an exception if an entity exists with the same name but it's not a directory
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise FileExistsError(
            f"{output_dir} exists, but it is not a directory. Consider removing it or renaming the output directory")


def train_neural_network(epochs, dataset, verbose=1):
    # imports
    from models import EmbeddingAndClassifier, LatentSiamesation
    from sklearn.metrics import f1_score, precision_score, recall_score
    from time import time
    import tensorflow as tf
    import tempfile
    import os
    import numpy as np

    # creating best weights file path
    embedding_weights_file = os.path.join("e2e", "best_weights.h5")
    prediction_model_file = os.path.join("e2e", "model.h5")
    siamese_weights_file = os.path.join(tempfile.gettempdir(), f"best_weights_{int(time() * 1000)}.h5")

    # creating the embedding and prediction model
    emb_model, pred_model = EmbeddingAndClassifier(dataset.max_event_number + 1)

    # creating the Siamese network
    print("Training the network")
    model = LatentSiamesation(emb_model)

    # training
    model.fit(
        # train data
        dataset.train_e2e_generator,
        steps_per_epoch=dataset.train_steps_per_epoch,

        # validation data
        validation_data=dataset.validation_e2e,

        # epochs
        epochs=epochs,

        # callbacks
        callbacks=[

            # saving best weights
            tf.keras.callbacks.ModelCheckpoint(siamese_weights_file,
                                               monitor="val_model_f1_score",
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
    model.load_weights(siamese_weights_file)

    # evaluating model
    print("Evaluating The Neural Network:")
    print("Test:")
    x, y = dataset.test_samples
    y_pred = np.round(pred_model.predict(x, batch_size=128))
    print("Precision is", precision_score(y, y_pred))
    print("Recall is", recall_score(y, y_pred))
    print("F1 score is", f1_score(y, y_pred))
    pred_model.save_weights(embedding_weights_file)
    pred_model.save(prediction_model_file)


def main():
    # imports
    from data import Dataset

    # hyper-parameters
    dataset_name = "hadoop"
    train_data_proportion = 0.90
    validation_data_proportion = 0.099
    batch_size = 256
    train_pair_generation_strategy = "sample"
    validation_pair_generation_strategy = "sample"
    test_pair_generation_strategy = "all"
    shuffler_buffer_size = 2 ** 33
    epochs = 256

    # preparing dataset
    dataset = Dataset(dataset_name=dataset_name,
                      train_proportion=train_data_proportion,
                      validation_proportion=validation_data_proportion,
                      batch_size=batch_size,
                      train_pair_generation_strategy=train_pair_generation_strategy,
                      validation_pair_generation_strategy=validation_pair_generation_strategy,
                      test_pair_generation_strategy=test_pair_generation_strategy,
                      shuffler_buffer_size=shuffler_buffer_size)

    # training neural network
    train_neural_network(epochs, dataset)


if __name__ == '__main__':
    main()
