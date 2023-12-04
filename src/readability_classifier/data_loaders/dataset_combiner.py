from datasets import Dataset, concatenate_datasets, load_from_disk


def load_datasets(paths: list[str]) -> list[Dataset]:
    """
    Loads the datasets from the specified paths.
    :param paths: The paths to the datasets.
    :return: The datasets.
    """
    datasets = []
    for path in paths:
        dataset = load_from_disk(path)
        datasets.append(dataset)

    return datasets


def remove_ambiguous_samples(dataset: Dataset) -> Dataset:
    """
    Removes the samples from the dataset that have an ambiguous readability score.
    Unambiguous readability scores are the 25% with the lowest and the 25% with the
    highest readability scores.
    :param dataset: The dataset.
    :return: The dataset without the ambiguous samples.
    """
    # Sort the dataset samples by readability score
    sorted_samples = sorted(dataset, key=lambda x: x["score"])

    # Calculate the number of samples to remove from the start and end
    num_samples = len(sorted_samples)
    num_to_remove = int(num_samples * 0.25)

    # Get the indices of samples to be kept (lowest 25% and highest 25%)
    indices_to_keep = set(range(num_to_remove)).union(
        set(range(num_samples - num_to_remove, num_samples))
    )

    # Create a new dataset without the ambiguous samples
    filtered_samples = [
        sample for i, sample in enumerate(sorted_samples) if i in indices_to_keep
    ]

    return dataset.from_list(filtered_samples)


if __name__ == "__main__":
    dataset_name = "dataset_with_names"
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

    datasets = load_datasets([dorn_path, bw_path, scalabrio_path])
    datasets = [remove_ambiguous_samples(dataset) for dataset in datasets]
    combined_dataset = concatenate_datasets(datasets)
    combined_dataset.save_to_disk(combined_path)

    # Print the name of the snippet with the lowest readability score
    min_sample = min(combined_dataset, key=lambda x: x["score"])
    print(f"Snippet with lowest readability score: {min_sample['name']}")
