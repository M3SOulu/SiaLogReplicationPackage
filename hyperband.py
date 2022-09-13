def create_output_directory():
    # imports
    import os

    # define output directory name
    output_dir = "hyperband"

    # create directory if it doesn't exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # raise an exception if an entity exists with the same name but it's not a directory
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        raise FileExistsError(
            f"{output_dir} exists, but it is not a directory. Consider removing it or renaming the output directory")


def hyperband(dataset, max_epochs):
    from kerastuner.tuners import Hyperband
    from models import SiameseHyperModel
    from time import time
    import os

    hyper_model = SiameseHyperModel(dataset.max_event_number + 1)
    project_name = f"hyperband_{int(time() * 1000)}"

    tuner = Hyperband(
        hyper_model,
        max_epochs=max_epochs,
        objective="val_loss",
        directory="hyperband",
        project_name=project_name
    )

    tuner.search_space_summary()
    tuner.search(
        dataset.train_pair_generator,
        steps_per_epoch=dataset.train_steps_per_epoch,
        validation_data=dataset.validation_pairs,
    )

    tuner.results_summary()

    for i, model in enumerate(tuner.get_best_models(5)):
        model.save(os.path.join("hyperband", project_name, f"model_{i}.hdf5"))


def main():
    # imports
    from data import Dataset

    # create output directory
    create_output_directory()

    # hyper-parameters
    dataset_name = "bgl"
    train_data_proportion = 0.9
    validation_data_proportion = 0.03
    batch_size = 64
    # batch_size = 128 * distribute_strategy.num_replicas_in_sync
    train_pair_generation_strategy = "sample"
    validation_pair_generation_strategy = "sample"
    test_pair_generation_strategy = "all"
    shuffler_buffer_size = 2 ** 33
    max_epochs = 256

    # preparing dataset
    dataset = Dataset(dataset_name=dataset_name, train_proportion=train_data_proportion,
                      validation_proportion=validation_data_proportion,
                      batch_size=batch_size,
                      train_pair_generation_strategy=train_pair_generation_strategy,
                      validation_pair_generation_strategy=validation_pair_generation_strategy,
                      test_pair_generation_strategy=test_pair_generation_strategy,
                      shuffler_buffer_size=shuffler_buffer_size)

    hyperband(dataset, max_epochs)


if __name__ == '__main__':
    main()
