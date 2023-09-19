from datasets import concatenate_datasets, load_from_disk


def combine_datasets(input_paths: list[str], output_path: str) -> None:
    """
    Combines the datasets from the specified paths into one dataset.
    :param input_paths: The paths to the datasets.
    :param output_path: The path to the combined dataset.
    """
    # Load the datasets
    datasets = []
    for path in input_paths:
        dataset = load_from_disk(path)
        datasets.append(dataset)

    # Combine the datasets
    combined_dataset = concatenate_datasets(datasets)

    # Save the combined dataset
    combined_dataset.save_to_disk(output_path)


if __name__ == "__main__":
    dataset_name = "dataset_not_splitted"
    dorn_path = (
        "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/"
        "DatasetDornJava/dataset/" + dataset_name
    )
    bw_path = (
        "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/DatasetBW/"
        + dataset_name
    )
    scalabrio_path = (
        "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/"
        "Dataset/Dataset/" + dataset_name
    )
    combined_path = (
        "C:/Users/lukas/Meine Ablage/Uni/{SoSe23/Masterarbeit/Datasets/Combined"
    )

    combine_datasets([dorn_path, bw_path, scalabrio_path], combined_path)
