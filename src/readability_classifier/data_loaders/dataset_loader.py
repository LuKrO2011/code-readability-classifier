from datasets import load_from_disk

krod_path = r"D:\PyCharm_Projects_D\styler2.0\dataset_1"

if __name__ == "__main__":
    # Load the dataset
    dataset = load_from_disk(krod_path)

    # Get the first sample
    sample = dataset[1]

    # Print the sample
    print(sample)
