import os
import unittest

from readability_classifier.encoders.dataset_utils import load_raw_dataset
from readability_classifier.encoders.image_encoder import VisualEncoder, _code_to_image
from src.readability_classifier.utils.utils import load_code
from tests.readability_classifier.utils.utils import DirTest

RES_DIR = os.path.join(os.path.dirname(__file__), "../../res/")
CODE_DIR = RES_DIR + "code_snippets/"


class TestVisualEncoder(DirTest):
    visual_encoder = VisualEncoder()

    @unittest.skip("Takes too long.")
    def test_encode_visual_dataset(self):
        data_dir = "res/raw_datasets/scalabrio"

        # Load raw data
        raw_data = load_raw_dataset(data_dir)

        # Encode raw data
        encoded_data = self.visual_encoder.encode_dataset(raw_data)

        # Check if encoded data is not empty
        assert len(encoded_data) > 0

        # # Show the first image
        # bytes_to_image(tensor_to_bytes(encoded_data[0]["image"]), "code.png")
        # from PIL import Image
        #
        # img = Image.open("code.png")
        # img.show()

    def test_encode_visual_text(self):
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
        encoded_code = self.visual_encoder.encode_text(code)

        # Check if encoded code is not empty
        assert len(encoded_code) > 0

    def test_code_to_image(self):
        # Create temporary directory
        output_file = os.path.join(self.output_dir, "out.png")

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

    def test_text_to_image(self):
        output_file = os.path.join(self.output_dir, "out.png")

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
        # buse_5 = """
        # // A method for counting
        # public void getNumber(){
        #     int count = 0;
        #     while(count < 10){
        #         count++;
        #     }
        # }
        # """

        # Convert the code to an image
        _code_to_image(buse_5, output=output_file)
        # _code_to_image(buse_5, output=output_file, width=500, height=500, css=
        # os.path.join(RES_DIR, "../..", "src", "res", "css",
        #              "towards_high_quality.css"), change_padding=True)

        # Check if the image was created successfully
        assert os.path.exists(output_file)

        # # Show the image
        # from PIL import Image
        # img = Image.open(output_file)
        # img.show()

    def test_code_to_image_towards(self):
        output_file = os.path.join(self.output_dir, "out.png")

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
