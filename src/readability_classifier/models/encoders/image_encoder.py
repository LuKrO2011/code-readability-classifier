import concurrent.futures
import logging
import os
import re
from tempfile import TemporaryDirectory

import cv2
import imgkit
import numpy as np
import torch
from PIL import Image
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import JavaLexer
from torch import Tensor

from readability_classifier.models.encoders.dataset_utils import (
    EncoderInterface,
    ReadabilityDataset,
)


class VisualEncoder(EncoderInterface):
    """
    A class for encoding code snippets as images.
    """

    def encode_dataset(self, unencoded_dataset: list[dict]) -> ReadabilityDataset:
        """
        Encodes the given dataset as images.
        :param unencoded_dataset: The unencoded dataset.
        :return: The encoded dataset.
        """
        encoded_dataset = []

        # Convert the list of dictionaries to a list of code snippet strings
        code_snippets = [sample["code_snippet"] for sample in unencoded_dataset]

        # Log the number of code snippets to encode
        logging.info(f"Image: Number of code snippets to encode: {len(code_snippets)}")

        # Encode the code snippets
        encoded_code_snippets = dataset_to_image_tensors(code_snippets)

        # Convert the list of encoded code snippets to a ReadabilityDataset
        for i in range(len(encoded_code_snippets)):
            encoded_dataset.append(
                {
                    "image": encoded_code_snippets[i],
                }
            )

        # Log the number of samples in the encoded dataset
        logging.info(f"Image: Encoding done. Number of samples: {len(encoded_dataset)}")

        return ReadabilityDataset(encoded_dataset)

    def encode_text(self, text: str) -> dict:
        """
        Encodes the given text as an image.
        :param text: The text to encode.
        :return: The encoded text as an image (in bytes).
        """
        # Encode the code snippet
        image = code_to_image_tensor(text)

        # Log successful encoding
        logging.info("Image: Encoding done.")

        return {
            "image": image,
        }


DEFAULT_OUT = "code.png"
DEFAULT_IN = "code.java"
DEFAULT_CSS = os.path.join(os.path.dirname(__file__), "../../../res/css/towards.css")
HEX_REGEX = r"#[0-9a-fA-F]{6}"


def _load_colors_from_css(file: str, hex_colors_regex: str = HEX_REGEX) -> set[str]:
    """
    Load the css file and return the colors in it using the regex
    :param file: path to the css file
    :param hex_colors_regex: regex to find the colors
    :return: list of colors
    """
    with open(file) as f:
        css_code = f.read()

    colors = re.findall(hex_colors_regex, css_code, re.MULTILINE)

    return set(colors)


def _convert_hex_to_rgba(hex_colors: set[str]) -> set[tuple[int, int, int, int]]:
    """
    Convert the hex colors to rgba
    :param hex_colors: set of hex colors
    :return: set of rgba colors
    """
    return {
        tuple(int(allowed_color.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)) + (255,)
        for allowed_color in hex_colors
    }


def _remove_blur(
    img: Image, width: int, height: int, allowed_colors: set[tuple[int, int, int, int]]
) -> Image:
    """
    Remove the blur from the image.
    Set all the pixels that are not in the allowed colors to the closest allowed color.
    :param img: The image
    :param width: The width of the image
    :param height: The height of the image
    :param allowed_colors: The allowed colors
    :return: The image without blur
    """
    for i in range(width):
        for j in range(height):
            if img.getpixel((i, j)) not in allowed_colors:
                closest_color = min(
                    allowed_colors,
                    key=lambda x: sum(
                        abs(i - j) for i, j in zip(x, img.getpixel((i, j)), strict=True)
                    ),
                )
                img.putpixel((i, j), closest_color)

    return img


def _change_padding(img: Image, new_padding: int = 6) -> Image:
    """
    Remove the padding from the image. The padding is white.
    :param img: The image
    :param new_padding: The new padding
    :return: The image without padding
    """
    img = img.convert("RGB")  # Convert the image to RGB mode

    # Get the dimensions of the image
    width, height = img.size

    # Get the image data as a list of tuples
    img_data = list(img.getdata())

    # Function to check if a pixel is white (RGB value of 255, 255, 255)
    def is_white(pixel):
        return pixel == (255, 255, 255)

    # Find the bounding box of non-white pixels
    left = width
    right = 0
    top = height
    bottom = 0

    for y in range(height):
        for x in range(width):
            pixel = img_data[y * width + x]
            if not is_white(pixel):
                left = min(left, x)
                right = max(right, x)
                top = min(top, y)
                bottom = max(bottom, y)

    # Add the new padding to the bounding box
    left = max(0, left - new_padding)
    right = min(width - 1, right + new_padding)
    top = max(0, top - new_padding)
    bottom = min(height - 1, bottom + new_padding)

    # Crop the image based on the bounding box
    img = img.crop((left, top, right + 1, bottom + 1))

    return img


def _code_to_image(code: str, output: str = DEFAULT_OUT, css: str = DEFAULT_CSS):
    """
    Convert the given Java code to a visualisation/image.
    :param code: The code
    :param output: The path to save the image
    :param css: The css to use for styling the code
    :return: The image
    """
    # Convert the code to html
    lexer = JavaLexer()
    formatter = HtmlFormatter()
    html = highlight(code, lexer, formatter)

    # Set the options for imgkit
    options = {
        "format": "png",
        "quality": "100",
        "encoding": "UTF-8",
        "quiet": "",
    }

    # Convert the html code to image
    imgkit.from_string(html, output, css=css, options=options)

    # Open the image
    img = Image.open(output)

    # Reduce the padding of the image
    img = _change_padding(img)

    # Save the image
    img.save(output)


def code_to_image_tensor(
    text: str,
    out_dir: str = None,
    width: int = 128,
    height: int = 128,
    css: str = DEFAULT_CSS,
) -> Tensor:
    """
    Convert the given Java code to a visualisation/image and load it as a tensor.
    :param text: The code to visualize
    :param out_dir: The directory where the image should be stored. If None, a temporary
        directory is created.
    :param width: The width of the image
    :param height: The height of the image
    :param css: The css to use for styling the code
    :return: The image of the code as a tensor
    """
    temp_dir = None
    if out_dir is None:
        # Create a temporary directory
        temp_dir = TemporaryDirectory()
        out_dir = temp_dir.name
    else:
        # Create the save directory, if it does not exist
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

    # Convert the code to an image
    image_file = os.path.join(out_dir, DEFAULT_OUT)
    _code_to_image(text, output=image_file, css=css, width=width, height=height)

    # Return the image as tensor 128x128x3 (RGB)
    image_as_tensor = _open_image_as_tensor(image_file)

    # Delete the temporary directory
    if temp_dir is not None:
        temp_dir.cleanup()

    return image_as_tensor


def _process_code_to_image(
    snippet: str, idx: int, save_dir: str, css: str, width: int, height: int
) -> None:
    """
    Process a single code snippet to an image. Used for parallel processing.
    :param snippet: The code snippet
    :param idx: The index of the code snippet
    :param save_dir: The directory where the image should be stored
    :param css: The css to use for styling the code
    :param width: The width of the image
    :param height: The height of the image
    :return: None
    """
    filename = os.path.join(save_dir, f"{idx}.png")
    _code_to_image(snippet, output=filename, css=css)


def dataset_to_image_tensors(
    snippets: list[str],
    save_dir: str = None,
    width: int = 128,
    height: int = 128,
    css: str = DEFAULT_CSS,
    parallel: bool = True,
) -> list[Tensor]:
    """
    Convert the given list with java code snippets to visualisations/images and load
    them as tensors.
    :param snippets: The list with java code snippets
    :param save_dir: The directory where the image should be stored.
    If None, a temporary directory is created.
    :param width: The width of the image
    :param height: The height of the image
    :param css: The css to use for styling the code
    :param parallel: Whether to use parallel processing
    :return: The images of the code snippets as tensors
    """
    temp_dir = None
    if save_dir is None:
        # Create a temporary directory
        temp_dir = TemporaryDirectory()
        save_dir = temp_dir.name
    else:
        # Create the save directory, if it does not exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

    if parallel:
        # Create the visualisations
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    _process_code_to_image, snippet, idx, save_dir, css, width, height
                )
                for idx, snippet in enumerate(snippets)
            }

            for future in concurrent.futures.as_completed(futures):
                future.result()
    else:
        # Create the visualisations
        for idx, snippet in enumerate(snippets):
            name = os.path.join(save_dir, f"{idx}.png")
            _code_to_image(snippet, output=name, css=css)

    # Read the images
    images_as_tensors = []
    for idx in range(len(snippets)):
        image_file = os.path.join(save_dir, f"{idx}.png")
        image_as_tensor = _open_image_as_tensor_2(
            image_file, width=width, height=height
        )
        images_as_tensors.append(image_as_tensor)

    # Delete the temporary directory
    if temp_dir is not None:
        temp_dir.cleanup()

    return images_as_tensors


def _open_image_as_tensor(image_path: str) -> Tensor:
    """
    Opens a png image as rgb tensor. Removes the alpha channel and transforms the values
    to float32. The shape of the tensor is (3, height, width).
    The images still have a blur or not 100% accurate colors.
    :param image_path: The path to the image
    :return: The image as a tensor
    """
    # Open the image using PIL
    img = Image.open(image_path)

    # Convert PIL image to NumPy array
    img_array = np.array(img)

    # Remove the alpha channel
    img_array = img_array[:, :, :3]

    # Transpose the array to get the shape (3, height, width)
    img_array = np.transpose(img_array, (2, 0, 1)) / 255

    # Convert NumPy array to tensor
    return torch.tensor(img_array, dtype=torch.float32)


def _open_image_as_tensor_2(image_path: str, width: int, height: int) -> Tensor:
    """
    Opens a png image as rgb tensor. Removes the alpha channel and transforms the values
    to float32. The shape of the tensor is (3, height, width).
    The images still have a blur or not 100% accurate colors.
    :param image_path: The path to the image
    :param width: The width of the image
    :param height: The height of the image
    :return: The image as a tensor
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (width, height))
    img_array = np.transpose(img, (2, 0, 1)) / 255
    return torch.tensor(img_array, dtype=torch.float32)
