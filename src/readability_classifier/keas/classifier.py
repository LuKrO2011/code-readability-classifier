import numpy as np

from readability_classifier.encoders.dataset_utils import (
    ReadabilityDataset,
)

# Define parameters
STATS_FILE_NAME = "stats.json"
DEFAULT_STORE_DIR = "output"


def convert_to_towards_input_without_score(
    encoded_data: ReadabilityDataset,
) -> dict[str, np.ndarray]:
    """
    Convert the encoded data to towards input without score.
    :param encoded_data: The encoded data.
    :return: The towards input.
    """
    x = encoded_data[0]

    encoded = {
        "struc_input": x["matrix"].numpy(),
        "vis_input": np.transpose(x["image"], (1, 2, 0)).numpy(),
        "seman_input_token": x["bert"]["input_ids"].numpy().squeeze(),
        "seman_input_segment": x["bert"]["segment_ids"].numpy().squeeze()
        if "segment_ids" in x["bert"]
        else x["bert"]["position_ids"].numpy().squeeze(),
    }

    # Convert to single numpy array
    return {k: np.array([v]) for k, v in encoded.items()}
