import os
from tempfile import TemporaryDirectory

import pytest

from readability_classifier.models.encoders.dataset_utils import load_raw_dataset
from readability_classifier.models.encoders.image_encoder import (
    VisualEncoder,
    _code_to_image,
)
from readability_classifier.utils.utils import load_code


@pytest.fixture()
def visual_encoder():
    return VisualEncoder()


@pytest.mark.skip()  # Disabled, because it takes too long
def test_encode_visual_dataset(visual_encoder):
    data_dir = "res/raw_datasets/scalabrio"

    # Load raw data
    raw_data = load_raw_dataset(data_dir)

    # Encode raw data
    encoded_data = visual_encoder.encode_dataset(raw_data)

    # Check if encoded data is not empty
    assert len(encoded_data) > 0

    # # Show the first image
    # bytes_to_image(tensor_to_bytes(encoded_data[0]["image"]), "code.png")
    # from PIL import Image
    #
    # img = Image.open("code.png")
    # img.show()


def test_encode_visual_text(visual_encoder):
    # Sample Java code
    code = """
    // A method for counting
    public void getNumber(){
        int count = 0;
        while(count < 10){
            count++;
        }
    }
    """

    # Encode the code
    encoded_code = visual_encoder.encode_text(code)

    # Check if encoded code is not empty
    assert len(encoded_code) > 0

    # Show the image
    # bytes_to_image(encoded_code, "code.png")
    # from PIL import Image
    # img = Image.open("code.png")
    # img.show()


RES_DIR = os.path.join(os.path.dirname(__file__), "../../../res/")
CODE_DIR = RES_DIR + "code_snippets/"


def test_code_to_image():
    # Create temporary directory
    temp_dir = TemporaryDirectory()
    output_file = os.path.join(temp_dir.name, "out.png")

    # Load the code
    filename = "AreaShop/AddCommand.java/execute.java"

    # Load the code
    code = load_code(CODE_DIR + filename)

    # Convert the code to an image
    _code_to_image(code, output=output_file)

    # Check if the image was created successfully
    assert os.path.exists(output_file)

    # # Show the image
    # from PIL import Image
    # img = Image.open(output_file)
    # img.show()

    # Clean up temporary directories
    temp_dir.cleanup()


def test_text_to_image():
    # Create temporary directory
    # temp_dir = TemporaryDirectory()
    # output_file = os.path.join(temp_dir.name, "out.png")

    output_file = "out.png"

    # Sample Java code
    buse_5 = """
    /**
     * Quits the application without any questions.
     */
    public void quit() {
        getConnectController().quitGame(true);
        if (!windowed) {
            gd.setFullScreenWindow(null);
        }
        System.exit(0);
    """

    # Convert the code to an image
    _code_to_image(buse_5, output=output_file)

    # Check if the image was created successfully
    assert os.path.exists(output_file)

    # # Show the image
    # from PIL import Image
    # img = Image.open(output_file)
    # img.show()

    # Clean up temporary directories
    # temp_dir.cleanup()


def test_code_to_image_towards():
    # Create temporary directory
    temp_dir = TemporaryDirectory()
    output_file = os.path.join(temp_dir.name, "out.png")

    # Load the code
    filename = "towards.java"

    # Load the code
    code = load_code(CODE_DIR + filename)

    # Convert the code to an image
    _code_to_image(code, output=output_file)

    # Check if the image was created successfully
    assert os.path.exists(output_file)

    # # Show the image
    # from PIL import Image
    # img = Image.open(output_file)
    # img.show()

    # Clean up temporary directories
    temp_dir.cleanup()
