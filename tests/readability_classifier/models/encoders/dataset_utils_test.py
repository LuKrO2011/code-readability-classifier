from readability_classifier.models.encoders.dataset_utils import load_encoded_dataset


def test_load_encoded_dataset():
    data_dir = "res/encoded_datasets/bw"

    # Load encoded data
    encoded_data = load_encoded_dataset(data_dir)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0
