import math
import os
import random
import re
from dataclasses import dataclass

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, regularizers
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.util import tf_inspect
from transformers import BertTokenizer


class BertConfig:
    """
    Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.vocab_size = kwargs.pop("vocab_size", 30000)
        self.type_vocab_size = kwargs.pop("type_vocab_size", 300)
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.num_hidden_layers = kwargs.pop("num_hidden_layers", 12)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 12)
        self.intermediate_size = kwargs.pop("intermediate_size", 3072)
        self.hidden_activation = kwargs.pop("hidden_activation", "gelu")
        self.hidden_dropout_rate = kwargs.pop("hidden_dropout_rate", 0.1)
        self.attention_dropout_rate = kwargs.pop("attention_dropout_rate", 0.1)
        self.max_position_embeddings = kwargs.pop("max_position_embeddings", 200)
        self.max_sequence_length = kwargs.pop("max_sequence_length", 200)


class BertEmbedding(keras.layers.Layer):
    """
    An own embedding layer that can be used for both token embeddings and
    segment embeddings in the BERT model.
    """

    def __init__(self, config, **kwargs):
        super().__init__(name="BertEmbedding")
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.token_embedding = self.add_weight(
            "weight",
            shape=[self.vocab_size, self.hidden_size],
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )
        self.type_vocab_size = config.type_vocab_size

        self.position_embedding = keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="position_embedding",
        )
        self.token_type_embedding = keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="token_type_embedding",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            epsilon=1e-12, name="LayerNorm"
        )
        self.dropout = keras.layers.Dropout(config.hidden_dropout_rate)

    def build(self, input_shape: tf.TensorShape):
        """
        Build the layer.
        :param input_shape: The shape of the input tensor.
        :return: None
        """
        with tf.name_scope("bert_embeddings"):
            super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False, mode: str = "embedding"):
        """
        Forward pass of the layer.
        """
        # used for masked lm
        if mode == "linear":
            return tf.matmul(inputs, self.token_embedding, transpose_b=True)

        # used for sentence classification
        input_ids, token_type_ids = inputs
        input_ids = tf.cast(input_ids, dtype=tf.int32)
        position_ids = tf.range(input_ids.shape[1], dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_ids.shape.as_list(), 0)

        # create embeddings
        position_embeddings = self.position_embedding(position_ids)
        token_type_embeddings = self.token_type_embedding(token_type_ids)
        token_embeddings = tf.gather(self.token_embedding, input_ids)

        # sum embeddings
        embeddings = token_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_config(self):
        """Returns the config of the layer.

        A layer config is a Python dictionary (serializable)
        containing the configuration of a layer.
        The same layer can be reinstantiated later
        (without its trained weights) from this configuration.

        The config of a layer does not include connectivity
        information, nor the layer class name. These are handled
        by `Network` (one layer of abstraction above).

        Returns:
            Python dictionary.
        """
        all_args = tf_inspect.getfullargspec(self.__init__).args
        config = {
            "name": self.name,
            "trainable": self.trainable,
        }
        if hasattr(self, "_batch_input_shape"):
            config["batch_input_shape"] = self._batch_input_shape
        # config['dtype'] = policy.serialize(self._dtype_policy)
        if hasattr(self, "dynamic"):
            # Only include `dynamic` in the `config` if it is `True`
            if self.dynamic:
                config["dynamic"] = self.dynamic
            elif "dynamic" in all_args:
                all_args.remove("dynamic")
        expected_args = config.keys()
        # Finds all arguments in the `__init__` that are not in the config:
        extra_args = [arg for arg in all_args if arg not in expected_args]
        # Check that either the only argument in the `__init__` is  `self`,
        # or that `get_config` has been overridden:
        if len(extra_args) > 1 and hasattr(self.get_config, "_is_default"):
            raise NotImplementedError(
                "Layer %s has arguments in `__init__` and "
                "therefore must override `get_config`." % self.__class__.__name__
            )
        return config


# Define the path of the data
STRUCTURE_DIR = "../../res/keras/Dataset/Processed Dataset/Structure"
TEXTURE_DIR = "../../res/keras/Dataset/Processed Dataset/Texture"
PICTURE_DIR = "../../res/keras/Dataset/Processed Dataset/Image"
MODEL_OUTPUT = "../../res/keras/Experimental output/towards_best.h5"

# Regex and patterns
JAVA_NAMING_REGEX = re.compile(r"([a-z]+)([A-Z]+)")
BRACKETS_AND_BACKSLASH = '["\\[\\]\\\\]'
SPECIAL_CHARACTERS = "[*.+!$#&,;{}()':=/<>%-]"
UNDERSCORE = "[_]"

# Define parameters
MAX_LEN = 100
TOTAL_SAMPLES = 210
TRAINING_SAMPLES = int(TOTAL_SAMPLES * 0.7)
VALIDATION_SAMPLES = TOTAL_SAMPLES - TRAINING_SAMPLES
MAX_WORDS = 1000
TOKENIZER_NAME = "bert-base-cased"

# Default values
DEFAULT_LEARNING_RATE = 0.0015
DEFAULT_LOSS = "binary_crossentropy"
DEFAULT_METRICS = [
    "acc",
    "Recall",
    "Precision",
    "AUC",
    "TruePositives",
    "TrueNegatives",
    "FalseNegatives",
    "FalsePositives",
]


class StructurePreprocessor:
    """
    Preprocessor for the structure data.
    """

    @classmethod
    def process(cls, structure_dir: str) -> tuple[list, dict, dict]:
        """
        Preprocess the structure data.
        :param structure_dir: The directory of the structure data.
        :return: The file names and the dictionary that stores the structure information
        """
        file_name = []
        data = {}
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
                    data[f_name.split(".")[0]] = 0
                else:
                    data[f_name.split(".")[0]] = 1
                data_structure[f_name.split(".")[0]] = lines

        return file_name, data, data_structure


class TexturePreprocessor:
    """
    Preprocessor for the texture data.
    """

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

    def process(
        self, texture_dir: str, max_len: int = MAX_LEN
    ) -> tuple[dict, dict, dict]:
        """
        Preprocess the texture data.

        :param texture_dir: The directory of the texture data.
        :param max_len: The maximum length of the text.
        :return: The dictionary that stores the token, position, and segment information
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

        return data_token, data_position, data_segment

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


class PicturePreprocessor:
    """
    Preprocessor for the picture data.
    """

    @staticmethod
    def process(picture_dir: str) -> tuple[dict, list]:
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

        return data_picture, data_image

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


def random_dataset(
    file_name: list,
    data_set: dict,
    data_structure: dict,
    data_picture: dict,
    data_token: dict,
    data_segment: dict,
    num_samples: int = 210,
) -> tuple[list, list, list, list, list, list]:
    """
    Randomly select num_samples samples from the dataset.
    :param file_name: The list of file names.
    :param data_set: The dictionary that stores the data.
    :param data_structure: The dictionary that stores the structure information.
    :param data_picture: The dictionary that stores the picture information.
    :param data_token: The dictionary that stores the token information.
    :param data_segment: The dictionary that stores the segment information.
    :param num_samples: The number of samples to select.
    :return: The randomly selected samples.
    """
    count_id = 0
    all_data = []
    label = []
    structure = []
    image = []
    token = []
    segment = []

    while count_id < num_samples and file_name:
        index_id = random.randint(0, len(file_name) - 1)
        item = file_name.pop(index_id)
        all_data.append(item)
        label.append(data_set[item])
        structure.append(data_structure[item])
        image.append(data_picture[item])
        token.append(data_token[item])
        segment.append(data_segment[item])
        count_id += 1

    return all_data, label, structure, image, token, segment


def create_classification_model(input_layer: tf.Tensor) -> tf.Tensor:
    """
    Create the classification model.
    :param input_layer: The input layer of the model.
    :return: The output layer of the model.
    """
    dense1 = layers.Dense(
        units=64, activation="relu", kernel_regularizer=regularizers.l2(0.001)
    )(input_layer)
    drop = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(units=16, activation="relu", name="random_detail")(drop)
    return layers.Dense(1, activation="sigmoid")(dense2)


def create_structural_extractor(
    input_shape: tuple[int, int] = (50, 305)
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Create the structural extractor layers.
    :param input_shape: The input shape of the model.
    :return: The input layer and the flattened layer.
    """
    model_input = layers.Input(shape=input_shape, name="structure")
    reshaped_input = layers.Reshape((*input_shape, 1))(model_input)

    conv1 = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(reshaped_input)
    pool1 = layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(pool1)
    pool2 = layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    conv3 = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(pool2)
    pool3 = layers.MaxPooling2D(pool_size=3, strides=3)(conv3)

    flattened = layers.Flatten()(pool3)
    return model_input, flattened


def create_structural_model(
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """
    Create the structural model for the matrix encoding.
    :return: The model.
    """

    structure_input, structure_flatten = create_structural_extractor()
    classification_output = create_classification_model(structure_flatten)

    model = models.Model(structure_input, classification_output)

    rms = optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=rms,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
    )

    return model


def create_semantic_extractor(
    input_shape: tuple[int, int] = (MAX_LEN,)
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Create the semantic extractor layers.
    :param input_shape: The input shape of the model.
    :return: The input layer, the token embedding layer, and the segment embedding layer
    """
    token_input = layers.Input(shape=input_shape, name="token")
    segment_input = layers.Input(shape=input_shape, name="segment")

    embedding = BertEmbedding(config=BertConfig(max_sequence_length=MAX_LEN))(
        [token_input, segment_input]
    )

    conv1 = layers.Conv1D(32, 5, activation="relu")(embedding)
    pool1 = layers.MaxPooling1D(3)(conv1)

    conv2 = layers.Conv1D(32, 5, activation="relu")(pool1)

    gru = layers.Bidirectional(layers.LSTM(32))(conv2)

    return token_input, segment_input, gru


def create_semantic_model(learning_rate: float = DEFAULT_LEARNING_RATE) -> keras.Model:
    """
    Create the semantic model for the bert encoding.
    :param learning_rate: The learning rate of the model.
    :return: The model.
    """
    token_input, segment_input, gru = create_semantic_extractor()
    classification_output = create_classification_model(gru)

    model = models.Model([token_input, segment_input], classification_output)

    rms = optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=rms,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
    )
    return model


def create_visual_extractor(
    input_shape: tuple[int, int, int] = (128, 128, 3)
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Create the visual extractor layers for the image encoding.
    :param input_shape: The input shape of the model.
    :return: The input layer and the flattened layer.
    """
    model_input = layers.Input(shape=input_shape, name="image")

    conv1 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(
        model_input
    )
    pool1 = layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    conv2 = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")(
        pool1
    )
    pool2 = layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    conv3 = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")(
        pool2
    )
    pool3 = layers.MaxPooling2D(pool_size=2, strides=2)(conv3)

    flattened = layers.Flatten()(pool3)
    return model_input, flattened


def create_visual_model(learning_rate: float = DEFAULT_LEARNING_RATE) -> keras.Model:
    """
    Create the visual model for the image encoding.
    :param learning_rate: The learning rate of the model.
    :return: The model.
    """
    image_input, image_flatten = create_visual_extractor()
    classification_output = create_classification_model(image_flatten)

    model = models.Model(image_input, classification_output)

    rms = optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=rms,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
    )
    return model


def create_towards_model(learning_rate: float = DEFAULT_LEARNING_RATE) -> keras.Model:
    """
    Create the VST model.
    :return: The model.
    """
    structure_input, structure_flatten = create_structural_extractor()
    token_input, segment_input, gru = create_semantic_extractor()
    image_input, image_flatten = create_visual_extractor()

    concatenated = layers.concatenate([structure_flatten, gru, image_flatten], axis=-1)

    dense1 = layers.Dense(
        units=64, activation="relu", kernel_regularizer=regularizers.l2(0.001)
    )(concatenated)
    drop = layers.Dropout(0.5)(dense1)
    dense2 = layers.Dense(units=16, activation="relu", name="random_detail")(drop)
    classification_output = layers.Dense(1, activation="sigmoid")(dense2)

    model = models.Model(
        [structure_input, token_input, segment_input, image_input],
        classification_output,
    )

    rms = optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=rms,
        loss=DEFAULT_LOSS,
        metrics=DEFAULT_METRICS,
    )
    return model


def get_from_dict(dictionary, key_start: str):
    """
    Get a value from a dict by key_start. The first value of the dict where the key
    starts with key_start is returned.
    :param dictionary: The dict to search in.
    :param key_start: The start of the key.
    :return:
    """
    for key, value in dictionary.items():
        if key.startswith(key_start):
            return value
    raise KeyError(f"Key {key_start} not found in dictionary")


def get_validation_set(
    fold_index: int,
    num_sample: int,
    train_structure: np.ndarray,
    train_token: np.ndarray,
    train_segment: np.ndarray,
    train_image: np.ndarray,
    train_label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the validation set.
    :param fold_index: The index of the fold.
    :param num_sample: The number of samples.
    :param train_structure: The training structure data.
    :param train_token: The training token data.
    :param train_segment: The training segment data.
    :param train_image: The training image data.
    :param train_label: The training label data.
    :return: The validation set.
    """
    val_indices = slice(fold_index * num_sample, (fold_index + 1) * num_sample)

    x_val_structure = train_structure[val_indices]
    x_val_token = train_token[val_indices]
    x_val_segment = train_segment[val_indices]
    x_val_image = train_image[val_indices]
    y_val = train_label[val_indices]

    return x_val_structure, x_val_token, x_val_segment, x_val_image, y_val


def get_training_set(
    fold_index: int,
    num_sample: int,
    train_structure: np.ndarray,
    train_token: np.ndarray,
    train_segment: np.ndarray,
    train_image: np.ndarray,
    train_label: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the training set.
    :param fold_index: The index of the fold.
    :param num_sample: The number of samples.
    :param train_structure: The training structure data.
    :param train_token: The training token data.
    :param train_segment: The training segment data.
    :param train_image: The training image data.
    :param train_label: The training label data.
    :return: The training set.
    """
    train_indices_1 = slice(0, fold_index * num_sample)
    train_indices_2 = slice((fold_index + 1) * num_sample, None)

    x_train_structure = np.concatenate(
        [train_structure[train_indices_1], train_structure[train_indices_2]], axis=0
    )
    x_train_token = np.concatenate(
        [train_token[train_indices_1], train_token[train_indices_2]], axis=0
    )
    x_train_segment = np.concatenate(
        [train_segment[train_indices_1], train_segment[train_indices_2]], axis=0
    )
    x_train_image = np.concatenate(
        [train_image[train_indices_1], train_image[train_indices_2]], axis=0
    )
    y_train = np.concatenate(
        [train_label[train_indices_1], train_label[train_indices_2]], axis=0
    )

    return x_train_structure, x_train_token, x_train_segment, x_train_image, y_train


@dataclass
class TowardsInput:
    """
    Data class for the input of the TowardsModel.
    """

    label: np.ndarray
    structure: np.ndarray
    image: np.ndarray
    token: np.ndarray
    segment: np.ndarray


# TODO: Remove computation of those: data_position and data_image (unused)
class Classifier:
    """
    A source code readability classifier.
    """

    label: np.ndarray = None
    structure: np.ndarray = None
    image: np.ndarray = None
    token: np.ndarray = None
    segment: np.ndarray = None

    def __init__(self, towards_model: keras.Model) -> None:
        """
        Initializes the classifier.
        :param towards_model: The towards model.
        """
        self.towards_model = towards_model

    def train(self):
        """
        Train the model.
        :return: None
        """
        file_name, data_set, data_structure = StructurePreprocessor.process(
            STRUCTURE_DIR
        )
        data_token, _, data_segment = TexturePreprocessor().process(TEXTURE_DIR)
        data_picture, _ = PicturePreprocessor.process(PICTURE_DIR)

        all_data, label, structure, image, token, segment = random_dataset(
            file_name=file_name,
            data_set=data_set,
            data_structure=data_structure,
            data_picture=data_picture,
            data_token=data_token,
            data_segment=data_segment,
        )

        # format the data
        self.label = np.asarray(label)
        self.structure = np.asarray(structure)
        self.image = np.asarray(image)
        self.token = np.asarray(token)
        self.segment = np.asarray(segment)

        print("Shape of structure data tensor:", self.structure.shape)
        print("Shape of image data tensor:", self.image.shape)
        print("Shape of token tensor:", self.token.shape)
        print("Shape of segment tensor:", self.segment.shape)
        print("Shape of label tensor:", self.label.shape)

        k_fold = 10
        num_sample = math.ceil(len(self.label) / k_fold)
        train_acc = []
        history = []

        for fold_index in range(k_fold):
            print(f"Now is fold {fold_index}")
            self.train_fold(fold_index=fold_index, num_sample=num_sample)

        # data analyze
        best_val_f1 = []
        best_val_auc = []
        best_val_mcc = []

        epoch_time_vst = 1
        for history_item in history:
            MCC_vst = []
            F1_vst = []
            history_dict = history_item.fold_stats
            val_acc_values = history_dict["val_acc"]
            val_recall_value = get_from_dict(history_dict, "val_recall")
            val_precision_value = get_from_dict(history_dict, "val_precision")
            val_auc_value = get_from_dict(history_dict, "val_auc")
            val_false_negatives = get_from_dict(history_dict, "val_false_negatives")
            val_false_positives = get_from_dict(history_dict, "val_false_positives")
            val_true_positives = get_from_dict(history_dict, "val_true_positives")
            val_true_negatives = get_from_dict(history_dict, "val_true_negatives")
            for i in range(20):
                tp = val_true_positives[i]
                tn = val_true_negatives[i]
                fp = val_false_positives[i]
                fn = val_false_negatives[i]
                if tp > 0 and tn > 0 and fn > 0 and fp > 0:
                    result_mcc = (tp * tn - fp * fn) / (
                        math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
                    )
                    MCC_vst.append(result_mcc)
                    result_precision = tp / (tp + fp)
                    result_recall = tp / (tp + fn)
                    result_f1 = (
                        2
                        * result_precision
                        * result_recall
                        / (result_precision + result_recall)
                    )
                    F1_vst.append(result_f1)
            train_acc.append(np.max(val_acc_values))
            best_val_f1.append(np.max(F1_vst))
            best_val_auc.append(np.max(val_auc_value))
            best_val_mcc.append(np.max(MCC_vst))
            print("Processing fold #", epoch_time_vst)
            print("------------------------------------------------")
            print("best accuracy score is #", np.max(val_acc_values))
            print("average recall score is #", np.mean(val_recall_value))
            print("average precision score is #", np.mean(val_precision_value))
            print("best f1 score is #", np.max(F1_vst))
            print("best auc score is #", np.max(val_auc_value))
            print("best mcc score is #", np.max(MCC_vst))
            print()
            print()
            epoch_time_vst = epoch_time_vst + 1

        print("Average vst model acc score", np.mean(train_acc))
        print("Average vst model f1 score", np.mean(best_val_f1))
        print("Average vst model auc score", np.mean(best_val_auc))
        print("Average vst model mcc score", np.mean(best_val_mcc))
        print()

    def train_fold(self, fold_index: int, num_sample: int) -> keras.callbacks.History:
        # Get the validation set
        (
            x_val_structure,
            x_val_token,
            x_val_segment,
            x_val_image,
            y_val,
        ) = get_validation_set(
            fold_index=fold_index,
            num_sample=num_sample,
            train_structure=self.structure,
            train_token=self.token,
            train_segment=self.segment,
            train_image=self.image,
            train_label=self.label,
        )
        # Get the training set
        (
            x_train_structure,
            x_train_token,
            x_train_segment,
            x_train_image,
            y_train,
        ) = get_training_set(
            fold_index=fold_index,
            num_sample=num_sample,
            train_structure=self.structure,
            train_token=self.token,
            train_segment=self.segment,
            train_image=self.image,
            train_label=self.label,
        )
        # Train the model
        towards_model = create_towards_model()
        checkpoint = ModelCheckpoint(
            MODEL_OUTPUT, monitor="val_acc", verbose=1, save_best_only=True, model="max"
        )
        callbacks = [checkpoint]
        return towards_model.fit(
            [x_train_structure, x_train_token, x_train_segment, x_train_image],
            y_train,
            epochs=20,
            batch_size=42,
            callbacks=callbacks,
            verbose=0,
            validation_data=(
                [x_val_structure, x_val_token, x_val_segment, x_val_image],
                y_val,
            ),
        )


def main():
    """
    Main function.
    :return: None
    """
    towards_model = create_towards_model()
    classifier = Classifier(towards_model)
    classifier.train()


if __name__ == "__main__":
    main()
