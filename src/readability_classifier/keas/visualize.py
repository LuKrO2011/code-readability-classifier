import os

import visualkeras
from keras.utils import plot_model

from src.readability_classifier.keas.model import create_towards_model


def ensure_directory_exists(directory: str):
    """
    Ensures that the specified directory exists. If it doesn't, creates it.

    :param directory: The directory path to check or create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def visualize_model_architecture(
    model, file_name="model_architecture.png", directory="visualizations"
):
    """
    Visualizes the model architecture using Keras' plot_model and saves it to a file.

    :param model: The Keras model to visualize.
    :param file_name: The name of the file where the visualization will be saved.
    Default is "model_architecture.png".
    :param directory: The directory where the file will be saved.
    Default is "visualizations".
    """
    ensure_directory_exists(directory)
    file_path = os.path.join(directory, file_name)

    # Plot the model architecture using Keras' plot_model and save it
    plot_model(model, to_file=file_path, show_shapes=True, show_layer_names=True)
    print(f"Keras model architecture saved to {file_path}")


def visualize_model_with_visualkeras(
    model, file_name="model_layered_view.png", directory="visualizations"
):
    """
    Visualizes the model architecture using VisualKeras and saves it to a file.

    :param model: The Keras model to visualize.
    :param file_name: The name of the file where the visualization will be saved.
    Default is "model_layered_view.png".
    :param directory: The directory where the file will be saved.
    Default is "visualizations".
    """
    ensure_directory_exists(directory)
    file_path = os.path.join(directory, file_name)

    # Visualize model using VisualKeras and save the image
    image = visualkeras.layered_view(model, legend=True)
    image.save(file_path)
    print(f"VisualKeras model layered view saved to {file_path}")


if __name__ == "__main__":
    # Create the model
    model = create_towards_model()

    filename = "model-more-layers"

    # Visualize and save the model architecture
    visualize_model_architecture(model, file_name=f"{filename}.png")

    # Visualize the model architecture using VisualKeras
    visualize_model_with_visualkeras(model, file_name=f"{filename}_visualkeras.png")

    # Log the number of parameters in the model
    model.summary()
