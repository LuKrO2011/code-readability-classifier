import keras
import tensorflow as tf
from keras import layers, models, optimizers, regularizers
from tensorflow.python.util import tf_inspect

from readability_classifier.keras.legacy_encoders import MAX_LEN

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
