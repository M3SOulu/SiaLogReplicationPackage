def store_hyperparameters(hparams):
    from utils import dump_dict
    print("HyperParameters")
    for k, v in hparams.items():
        print(k, ":", v)
    with open(f"out/HyperParameters.txt", "w") as file:
        file.write(dump_dict(hparams))


def create_output_directory(directory_extension=None):
    # imports
    import os

    # define output directory name
    output_dir = "out" if directory_extension is None else f"out_{directory_extension}"

    # create directory if it doesn't exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # raise an exception if an entity exists with the same name but it's not a directory
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise FileExistsError(
            f"{output_dir} exists, but it is not a directory. Consider removing it or renaming the output directory")

    return output_dir


def dump_dataset_info(dataset):
    print("Positive sample count:", dataset.train_positive_sample_count)
    print("Negative sample count:", dataset.train_negative_sample_count)
    print("Train pair count:", dataset.train_pairs_count)
    print("Validation Pair count:", dataset.validation_pairs_count)
    print("Test Pair count:", dataset.test_pairs_count)
    print("Events count:", dataset.max_event_number)


def train_neural_network(epochs, dataset, output_directory, verbose=1, load_if_model_file_exists=True,
                         use_validation=True,
                         evaluate_embedding_model_after_training=True):
    # imports
    from models import embedding_hdfs, embedding_bgl, embedding_hadoop, SiameseNetwork
    from time import time
    from utils import dump_dict
    import tensorflow as tf
    import tempfile
    import os

    # creating best weights file path
    embedding_weights_file = os.path.join("out", f"best_weights_{dataset.name}.h5")
    siamese_weights_file = os.path.join(tempfile.gettempdir(), f"best_weights_{int(time() * 1000)}.h5")
    output_dict = {}

    # creating the embedding model and loading previously trained weights if possible
    emb_models = {"hdfs": embedding_hdfs, "bgl": embedding_bgl, "hadoop": embedding_hadoop}
    emb_model = emb_models[dataset.name](dataset.max_event_number + 1)
    print("Model summary:")
    emb_model.summary()
    if load_if_model_file_exists and os.path.exists(embedding_weights_file):
        print(f"Loading weights from {embedding_weights_file} instead of training from scratch")
        try:
            emb_model.load_weights(embedding_weights_file)
        except (ImportError, ValueError) as e:
            print(f"Error happened while loading model: {e}")

    # creating the Siamese network
    model = SiameseNetwork(emb_model)

    # train the model if it is not loaded
    if not load_if_model_file_exists:
        # training
        print("Training the network")
        train_history = model.fit(
            # train data
            dataset.train_pair_generator,
            steps_per_epoch=dataset.train_steps_per_epoch,

            # validation data
            validation_data=dataset.validation_pairs if use_validation else None,

            # epochs
            epochs=epochs,

            # callbacks
            callbacks=[

                # saving best weights
                tf.keras.callbacks.ModelCheckpoint(siamese_weights_file,
                                                   monitor="loss" if use_validation else "loss",
                                                   save_weights_only=True,
                                                   save_best_only=True,
                                                   verbose=verbose),

                # reducing learning rate on loss plateau
                tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", verbose=verbose),
            ],

            # verbose type
            verbose=verbose)
        output_dict["Train history"] = {f"Epoch {i + 1}": loss for i, loss in enumerate(train_history.history["loss"])}

        # loading the best neural network's weights
        print("Loading the best neural network's weights")
        model.load_weights(siamese_weights_file)

    # evaluating model
    if evaluate_embedding_model_after_training:
        print("Evaluating the neural network on the test data:")
        eval_result = model.evaluate(*dataset.test_pairs, batch_size=dataset.batch_size)
        output_dict["Test loss"] = eval_result

    # save the weights of the embedding model if it is not a loaded model
    if not load_if_model_file_exists:
        print("Saving the embedding model's weight")
        emb_model.save_weights(embedding_weights_file)

    with open(f"{output_directory}/SiameseNetTraining.txt", "w") as file:
        file.write(dump_dict(output_dict))

    return emb_model


def distribution_drift_monitor(emb_model, dataset, noise_ratio_interval=0.03,
                               noisy_datasets_per_ratio=5):
    # imports
    from distribution_monitors import GaussianDistributionMonitor
    import numpy as np

    # generate noisy data tables
    noisy_data_tables = {ratio: [dataset.create_noisy_data_table(ratio) for _ in range(noisy_datasets_per_ratio)] for
                         ratio in np.arange(0.0, 0.61, noise_ratio_interval)}

    # create distribution monitor
    monitor = GaussianDistributionMonitor(emb_model)
    monitor.fit(dataset.train_data_table)

    # producing resutls
    results = {ratio: np.mean([monitor.fitness_score(dt) for dt in data_tables]) for ratio, data_tables in
               noisy_data_tables.items()}

    return results


def evaluate_classifiers(emb_model, dataset, output_directory, verbose=1, noise_ratio_interval=0.05,
                         noisy_datasets_per_ratio=3):
    # imports
    from classifiers import LogisticRegression, KNearestNeighbors, SupportVectorMachine, \
        NeuralNetwork
    from utils import dump_dict, average_dict
    import numpy as np

    # generate noisy data tables
    noisy_data_tables = {ratio: [dataset.create_noisy_data_table(ratio) for _ in range(noisy_datasets_per_ratio)] for
                         ratio in np.arange(noise_ratio_interval, 0.31, noise_ratio_interval)}

    # create all results object
    all_results = {}

    # evaluations
    for classifier_type in [LogisticRegression, KNearestNeighbors, SupportVectorMachine, NeuralNetwork]:

        # create result object
        classifier_results = {}

        # create classifier object
        classifier = classifier_type(emb_model)

        # training classifier
        classifier.fit(dataset.train_data_table)

        # train set evaluation
        classifier_results["Train"] = classifier.evaluate(dataset.train_data_table)

        # test set evaluation
        classifier_results["Test"] = classifier.evaluate(dataset.test_data_table)

        # noisy test set evaluation
        classifier_results["Noisy input"] = {}
        for ratio, data_tables in noisy_data_tables.items():
            avg_results = average_dict([classifier.evaluate(data_table) for data_table in data_tables])
            classifier_results["Noisy input"][f"Ratio {ratio:.2f}"] = avg_results

        # storing the results
        all_results[classifier_type.__name__] = classifier_results

        # printing evaluations results in stdout
        if verbose:
            print(f"{classifier_type.__name__}:")
            print(dump_dict(classifier_results, indents=1))

        # writing evaluations results in output directory
        with open(f"{output_directory}/{classifier_type.__name__}.txt", "w") as output_file:
            output_file.write(dump_dict(classifier_results))
    return all_results


def export_visualizations(emb_model, dataset, output_directory):
    # imports
    from visualizers import SequenceEmbeddingProjectorVisualizer, EventEmbeddingProjectorVisualizer

    # visualizations
    SequenceEmbeddingProjectorVisualizer(emb_model, dataset).export()
    EventEmbeddingProjectorVisualizer(emb_model, dataset).export()


def export_plots(output_directory, eval_result=None, drift_result=None):
    # imports
    from plots import plot_noisy_input, plot_distribution_drift

    # plots
    if eval_result:
        plot_noisy_input(eval_result, f"{output_directory}/noisy_input_plot.pdf")

    if drift_result:
        plot_distribution_drift(drift_result, f"{output_directory}/distribution_drift_plot.pdf",
                                "out/distribution_drift_hadoop.csv")


def main():
    # imports
    from data import Dataset
    # from models import distribute_strategy

    # hyper-parameters
    dataset_name = "hadoop"
    train_data_proportion = 0.87
    validation_data_proportion = 0.03
    use_validation_on_training = False
    evaluate_emb_model_after_training = False
    batch_size = 128
    # batch_size = 128 * distribute_strategy.num_replicas_in_sync
    train_pair_generation_strategy = "all"
    validation_pair_generation_strategy = "sample"
    test_pair_generation_strategy = "all"
    k_negative_sample = 10
    shuffler_buffer_size = 2 ** 33
    epochs = 64
    load_if_model_file_exists = True
    noisy_datasets_per_ratio = 5

    # print info
    store_hyperparameters({name: value for name, value in locals().items() if name[0] != "_"})

    # create output directory
    output_directory = create_output_directory(dataset_name)

    # preparing dataset
    dataset = Dataset(dataset_name=dataset_name,
                      train_proportion=train_data_proportion,
                      validation_proportion=validation_data_proportion,
                      batch_size=batch_size,
                      train_pair_generation_strategy=train_pair_generation_strategy,
                      validation_pair_generation_strategy=validation_pair_generation_strategy,
                      test_pair_generation_strategy=test_pair_generation_strategy,
                      k_negative_samples=k_negative_sample,
                      shuffler_buffer_size=shuffler_buffer_size)

    # print dataset info
    dump_dataset_info(dataset)

    # training neural network
    emb_model = train_neural_network(epochs, dataset,
                                     output_directory=output_directory,
                                     load_if_model_file_exists=load_if_model_file_exists,
                                     use_validation=use_validation_on_training,
                                     evaluate_embedding_model_after_training=evaluate_emb_model_after_training)

    # evaluating classifiers
    print("Evaluating classifiers")
    eval_results = evaluate_classifiers(emb_model, dataset,
                                        output_directory=output_directory,
                                        verbose=1,
                                        noisy_datasets_per_ratio=noisy_datasets_per_ratio)

    # distribution drift test
    drift_result = distribution_drift_monitor(emb_model, dataset, noisy_datasets_per_ratio=noisy_datasets_per_ratio)

    # exporting visualization files
    print("Exporting visualization files")
    export_visualizations(emb_model, dataset, output_directory)

    # showing plots
    export_plots(output_directory, eval_results, drift_result)


if __name__ == '__main__':
    main()
