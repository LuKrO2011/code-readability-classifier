import os
import re
from dataclasses import dataclass

import cv2
import numpy as np
from transformers import BertTokenizer

# Default paths
STRUCTURE_DIR = "../../res/keras/Dataset/Processed Dataset/Structure"
TEXTURE_DIR = "../../res/keras/Dataset/Processed Dataset/Texture"
PICTURE_DIR = "../../res/keras/Dataset/Processed Dataset/Image"

# Regex and patterns
JAVA_NAMING_REGEX = re.compile(r"([a-z]+)([A-Z]+)")
BRACKETS_AND_BACKSLASH = '["\\[\\]\\\\]'
SPECIAL_CHARACTERS = "[*.+!$#&,;{}()':=/<>%-]"
UNDERSCORE = "[_]"

# Constants
MAX_LEN = 100
TOKENIZER_NAME = "bert-base-cased"


# WARNING: THIS CODE IS NOT USED ANYMORE
# Remove computation of those: data_position and data_image (unused)


def preprocess_data() -> list[dict]:
    """
    Load and preprocess the data.
    :return: The towards inputs.
    """
    structure_input = StructurePreprocessor.process(STRUCTURE_DIR)
    texture_input = TexturePreprocessor().process(TEXTURE_DIR)
    picture_input = PicturePreprocessor.process(PICTURE_DIR)

    # Combine the data into towards input
    return [
        {
            "label": structure_input[key].score,
            "structure": structure_input[key].lines,
            "image": picture_input[key].image,
            "token": texture_input[key].token,
            "segment": texture_input[key].segment,
        }
        for key in structure_input
    ]


@dataclass
class StructureInput:
    """
    Data class for the input of the StructuralModel (matrix encoding).
    """

    score: int
    lines: np.ndarray


class StructurePreprocessor:
    """
    Preprocessor for the structure data.
    """

    @classmethod
    def process(cls, structure_dir: str) -> dict[str, StructureInput]:
        """
        Preprocess the structure data.
        :param structure_dir: The directory of the structure data.
        :return: The dictionary that stores the structure information.
        """
        file_name = []
        score = {}
        data_structure = {}

        for label_type in ["Readable", "Unreadable"]:
            dir_name = os.path.join(structure_dir, label_type)
            for f_name in os.listdir(dir_name):
                with open(os.path.join(dir_name, f_name), errors="ignore") as f:
                    lines = []
                    if not f_name.startswith("."):
                        file_name.append(f_name.split(".")[0])
                        for line in f:
                            line = line.strip(",\n")
                            info = line.split(",")
                            info_int = []
                            count = 0
                            for item in info:
                                if count < 305:
                                    info_int.append(int(item))
                                    count += 1
                            info_int = np.asarray(info_int)
                            lines.append(info_int)
                lines = np.asarray(lines)
                if label_type == "Readable":
                    score[f_name.split(".")[0]] = 0
                else:
                    score[f_name.split(".")[0]] = 1
                data_structure[f_name.split(".")[0]] = lines

        # Turn the data into a dictionary of StructureInput objects
        return {
            key: StructureInput(score=score[key], lines=np.asarray(lines))
            for key, lines in data_structure.items()
        }


@dataclass
class TextureInput:
    """
    Data class for the input of the SemanticModel (BERT encoding).
    """

    score: int
    token: np.ndarray
    position: np.ndarray
    segment: np.ndarray


class TexturePreprocessor:
    """
    Preprocessor for the texture data.
    """

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    def process(
        self, texture_dir: str, max_len: int = MAX_LEN
    ) -> dict[str, TextureInput]:
        """
        Preprocess the texture data.

        :param texture_dir: The directory of the texture data.
        :param max_len: The maximum length of the text.
        :return: The dictionary that stores the texture information.
        """
        data_token = {}
        data_position = {}
        data_segment = {}

        # Process files in different label types ("Readable", "Unreadable")
        for label_type in ["Readable", "Unreadable"]:
            string_content = self._process_files_in_directory(
                os.path.join(texture_dir, label_type), max_len
            )
            self._process_string(
                string_content, data_token, data_position, data_segment, max_len
            )

        # Turn the data into a dictionary of TextureInput objects
        return {
            key: TextureInput(
                score=0 if key.startswith("Readable") else 1,
                token=np.asarray(data_token[key]),
                position=np.asarray(data_position[key]),
                segment=np.asarray(data_segment[key]),
            )
            for key in data_token
        }

    @classmethod
    def _process_files_in_directory(cls, directory: str, max_len: int) -> dict:
        """
        Process text files in a directory.

        :param directory: The directory path containing text files.
        :param max_len: The maximum length of the text.
        :return: A dictionary with processed string content.
        """
        string_content = {}

        for file_name in os.listdir(directory):
            if file_name.endswith(".txt"):
                content = cls._process_file(os.path.join(directory, file_name), max_len)
                string_content[file_name.split(".")[0]] = content

        return string_content

    @classmethod
    def _process_file(cls, file_path: str, max_len: int) -> str:
        """
        Process content in a text file.

        :param file_path: The path to the text file.
        :param max_len: The maximum length of the text.
        :return: Processed string content.
        """
        processed_content = ""
        with open(file_path, errors="ignore") as file:
            for content in file:
                content = re.sub(JAVA_NAMING_REGEX, r"\1 \2", content)
                content = re.sub(
                    BRACKETS_AND_BACKSLASH, lambda x: " " + x.group(0) + " ", content
                )
                content = re.sub(
                    SPECIAL_CHARACTERS, lambda x: " " + x.group(0) + " ", content
                )
                content = re.sub(UNDERSCORE, lambda x: " ", content)
                processed_content += cls._process_content(content, max_len)

        return processed_content

    @staticmethod
    def _process_content(content: str, max_len: int) -> str:
        """
        Process individual content and return processed string.

        :param content: Individual content to process.
        :param max_len: The maximum length of the text.
        :return: Processed string.
        """
        processed_string = ""
        count = 0
        for word in content.split():
            if len(word) > 1 or not word.isalpha():
                processed_string += " " + word
                count += 1
        while count < max_len:
            processed_string += " 0"  # Assuming "0" represents padding
            count += 1

        return processed_string

    def _process_string(
        self,
        string_content: dict,
        data_token: dict,
        data_position: dict,
        data_segment: dict,
        max_len: int,
    ) -> None:
        """
        Process string content to tokens and store in dictionaries.

        :param string_content: Dictionary with string content.
        :param data_token: Dictionary to store token information.
        :param data_position: Dictionary to store position information.
        :param data_segment: Dictionary to store segment information.
        :param max_len: The maximum length of the text.
        """
        for sample, content in string_content.items():
            list_token = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(content)
            )
            list_token = list_token[:max_len]
            while len(list_token) < max_len:
                list_token.append(0)
            data_token[sample] = list_token

            list_position = list(range(min(len(list_token), max_len)))
            data_position[sample] = list_position

            list_segment = list(range(len(list_position)))
            data_segment[sample] = list_segment


@dataclass
class PictureInput:
    """
    Data class for the input of the VisualModel (image encoding).
    """

    score: int
    image: np.ndarray


class PicturePreprocessor:
    """
    Preprocessor for the picture data.
    """

    @staticmethod
    def process(picture_dir: str) -> dict[str, PictureInput]:
        """
        Preprocess the picture data.
        :param picture_dir: The directory of the picture data.
        :return: The dictionary that stores the picture information.
        """
        data_picture = {}
        data_image = []

        for label_type in ["readable", "unreadable"]:
            dir_image_name = os.path.join(picture_dir, label_type)
            picture_dict, image_list = PicturePreprocessor.process_images(
                dir_image_name
            )
            data_picture.update(picture_dict)
            data_image.extend(image_list)

        # Turn the data into a dictionary of PictureInput objects
        return {
            key: PictureInput(
                score=0 if key.startswith("readable") else 1,
                image=np.asarray(data_picture[key]),
            )
            for key in data_picture
        }

    @staticmethod
    def process_images(directory: str) -> tuple[dict, list]:
        """
        Process images within a directory.
        :param directory: Path to the directory containing images.
        :return: A dictionary containing processed images and a list of images.
        """
        picture_data = {}
        image_data = []

        for f_name in os.listdir(directory):
            if not f_name.startswith("."):
                img_data = cv2.imread(os.path.join(directory, f_name))
                img_data = cv2.resize(img_data, (128, 128))
                result = img_data / 255.0
                picture_data[f_name.split(".")[0]] = result
                image_data.append(result)

        return picture_data, image_data
