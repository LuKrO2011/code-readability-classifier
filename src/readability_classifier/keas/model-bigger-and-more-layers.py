import keras
import tensorflow as tf
from keras import layers, models, optimizers, regularizers

from src.readability_classifier.keas.legacy_encoders import MAX_LEN

# ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)        ┃ Output Shape      ┃    Param # ┃ Connected to      ┃
# ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
# │ struc_input         │ (None, 50, 305)   │          0 │ -                 │
# │ (InputLayer)        │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_reshape       │ (None, 50, 305,   │          0 │ struc_input[0][0] │
# │ (Reshape)           │ 1)                │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_conv1         │ (None, 50, 305,   │        640 │ struc_reshape[0]… │
# │ (Conv2D)            │ 64)               │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_input           │ (None, 128, 128,  │          0 │ -                 │
# │ (InputLayer)        │ 3)                │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_conv2         │ (None, 50, 305,   │     36,928 │ struc_conv1[0][0] │
# │ (Conv2D)            │ 64)               │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_conv1 (Conv2D)  │ (None, 128, 128,  │      1,792 │ vis_input[0][0]   │
# │                     │ 64)               │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_pool1         │ (None, 25, 152,   │          0 │ struc_conv2[0][0] │
# │ (MaxPooling2D)      │ 64)               │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_input_token   │ (None, 100)       │          0 │ -                 │
# │ (InputLayer)        │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_input_segment │ (None, 100)       │          0 │ -                 │
# │ (InputLayer)        │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_pool1           │ (None, 64, 64,    │          0 │ vis_conv1[0][0]   │
# │ (MaxPooling2D)      │ 64)               │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_conv3         │ (None, 25, 152,   │     73,856 │ struc_pool1[0][0] │
# │ (Conv2D)            │ 128)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_bert          │ (None, 100, 768)  │ 23,425,536 │ seman_input_toke… │
# │ (BertEmbedding)     │                   │            │ seman_input_segm… │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_conv2 (Conv2D)  │ (None, 64, 64,    │     73,856 │ vis_pool1[0][0]   │
# │                     │ 128)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_conv4         │ (None, 25, 152,   │    147,584 │ struc_conv3[0][0] │
# │ (Conv2D)            │ 128)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_conv1         │ (None, 96, 64)    │    245,824 │ seman_bert[0][0]  │
# │ (Conv1D)            │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_pool2           │ (None, 32, 32,    │          0 │ vis_conv2[0][0]   │
# │ (MaxPooling2D)      │ 128)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_pool2         │ (None, 12, 76,    │          0 │ struc_conv4[0][0] │
# │ (MaxPooling2D)      │ 128)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_pool1         │ (None, 32, 64)    │          0 │ seman_conv1[0][0] │
# │ (MaxPooling1D)      │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_conv3 (Conv2D)  │ (None, 32, 32,    │    295,168 │ vis_pool2[0][0]   │
# │                     │ 256)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_conv5         │ (None, 12, 76,    │    819,456 │ struc_pool2[0][0] │
# │ (Conv2D)            │ 256)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_conv2         │ (None, 28, 128)   │     41,088 │ seman_pool1[0][0] │
# │ (Conv1D)            │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_pool3           │ (None, 16, 16,    │          0 │ vis_conv3[0][0]   │
# │ (MaxPooling2D)      │ 256)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_conv6         │ (None, 12, 76,    │  1,638,656 │ struc_conv5[0][0] │
# │ (Conv2D)            │ 256)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_pool2         │ (None, 9, 128)    │          0 │ seman_conv2[0][0] │
# │ (MaxPooling1D)      │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_conv4 (Conv2D)  │ (None, 16, 16,    │  1,180,160 │ vis_pool3[0][0]   │
# │                     │ 512)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_pool3         │ (None, 6, 38,     │          0 │ struc_conv6[0][0] │
# │ (MaxPooling2D)      │ 256)              │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_conv3         │ (None, 5, 256)    │    164,096 │ seman_pool2[0][0] │
# │ (Conv1D)            │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_pool4           │ (None, 8, 8, 512) │          0 │ vis_conv4[0][0]   │
# │ (MaxPooling2D)      │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ struc_flatten       │ (None, 58368)     │          0 │ struc_pool3[0][0] │
# │ (Flatten)           │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ seman_gru           │ (None, 128)       │    164,352 │ seman_conv3[0][0] │
# │ (Bidirectional)     │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ vis_flatten         │ (None, 32768)     │          0 │ vis_pool4[0][0]   │
# │ (Flatten)           │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ concatenate         │ (None, 91264)     │          0 │ struc_flatten[0]… │
# │ (Concatenate)       │                   │            │ seman_gru[0][0],  │
# │                     │                   │            │ vis_flatten[0][0] │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ class_dense1        │ (None, 64)        │  5,840,960 │ concatenate[0][0] │
# │ (Dense)             │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ class_dropout1      │ (None, 64)        │          0 │ class_dense1[0][… │
# │ (Dropout)           │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ class_dense2        │ (None, 32)        │      2,080 │ class_dropout1[0… │
# │ (Dense)             │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ class_dropout2      │ (None, 32)        │          0 │ class_dense2[0][… │
# │ (Dropout)           │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ class_dense3        │ (None, 16)        │        528 │ class_dropout2[0… │
# │ (Dense)             │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ class_dropout3      │ (None, 16)        │          0 │ class_dense3[0][… │
# │ (Dropout)           │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ class_dense4        │ (None, 8)         │        136 │ class_dropout3[0… │
# │ (Dense)             │                   │            │                   │
# ├─────────────────────┼───────────────────┼────────────┼───────────────────┤
# │ class_output        │ (None, 1)         │          9 │ class_dense4[0][… │
# │ (Dense)             │                   │            │                   │
# └─────────────────────┴───────────────────┴────────────┴───────────────────┘
# Total params: 34,152,705 (130.28 MB)
# Trainable params: 34,152,705 (130.28 MB)
# Non-trainable params: 0 (0.00 B)

# Freeze
# 2024-09-28 00:11:47,129,129 root INFO Overall results:
# Best validation acc score: 0.8511836027713626
# Best validation precision score: 0.7841041134092493
# Best validation recall score: 0.9706559263521288
# Best validation f1 score: 0.8674636842781848
# Best validation auc score: 0.8773800198806891
# Best validation mcc score: 0.7230615516244412
# Average validation acc score: 0.5769540675090377
# Average validation precision score: 0.4087200778140101
# Average validation recall score: 0.6958936723787088
# Average validation f1 score: 0.5098490186646681
# Average validation auc score: 0.5523068750963595

# No Freeze
# 2024-10-02 07:08:08,402,402 root INFO Overall results:
# Best validation acc score: 0.9047619047619048
# Best validation precision score: 0.8888888888888888
# Best validation recall score: 0.8888888888888888
# Best validation f1 score: 0.8888888888888888
# Best validation auc score: 0.8888888888888888
# Best validation mcc score: 0.8055555555555556
# Average validation acc score: 0.8047619047619048
# Average validation precision score: 0.8271252113899173
# Average validation recall score: 0.7904545454545455
# Average validation f1 score: 0.793442418306302
# Average validation auc score: 0.8087898784222313


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


def create_classification_layers(input_layer: tf.Tensor) -> tf.Tensor:
    """
    Create the classification model.
    :param input_layer: The input layer of the model.
    :return: The output layer of the model.
    """
    dense1 = layers.Dense(
        units=64,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.001),
        name="class_dense1",
    )(input_layer)
    drop1 = layers.Dropout(0.5, name="class_dropout1")(dense1)
    dense2 = layers.Dense(units=32, activation="relu", name="class_dense2")(drop1)
    drop2 = layers.Dropout(0.5, name="class_dropout2")(dense2)
    dense3 = layers.Dense(units=16, activation="relu", name="class_dense3")(drop2)
    drop3 = layers.Dropout(0.5, name="class_dropout3")(dense3)
    dense4 = layers.Dense(units=8, activation="relu", name="class_dense4")(drop3)
    return layers.Dense(1, activation="sigmoid", name="class_output")(dense4)


def create_structural_extractor(
    input_shape: tuple[int, int] = (50, 305)
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Create the structural extractor layers.
    :param input_shape: The input shape of the model.
    :return: The input layer and the flattened layer.
    """
    model_input = layers.Input(shape=input_shape, name="struc_input")
    reshaped_input = layers.Reshape((*input_shape, 1), name="struc_reshape")(
        model_input
    )

    # First convolutional block
    conv1 = layers.Conv2D(
        filters=64, kernel_size=3, activation="relu", padding="same", name="struc_conv1"
    )(reshaped_input)
    conv2 = layers.Conv2D(
        filters=64, kernel_size=3, activation="relu", padding="same", name="struc_conv2"
    )(conv1)
    pool1 = layers.MaxPooling2D(pool_size=2, strides=2, name="struc_pool1")(conv2)

    # Second convolutional block
    conv3 = layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same",
        name="struc_conv3",
    )(pool1)
    conv4 = layers.Conv2D(
        filters=128,
        kernel_size=3,
        activation="relu",
        padding="same",
        name="struc_conv4",
    )(conv3)
    pool2 = layers.MaxPooling2D(pool_size=2, strides=2, name="struc_pool2")(conv4)

    # Third convolutional block
    conv5 = layers.Conv2D(
        filters=256,
        kernel_size=5,
        activation="relu",
        padding="same",
        name="struc_conv5",
    )(pool2)
    conv6 = layers.Conv2D(
        filters=256,
        kernel_size=5,
        activation="relu",
        padding="same",
        name="struc_conv6",
    )(conv5)
    pool3 = layers.MaxPooling2D(pool_size=2, strides=2, name="struc_pool3")(conv6)

    # Flatten the output
    flattened = layers.Flatten(name="struc_flatten")(pool3)

    return model_input, flattened


def create_structural_model(
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """
    Create the structural model for the matrix encoding.
    :return: The model.
    """

    structure_input, structure_flatten = create_structural_extractor()
    classification_output = create_classification_layers(structure_flatten)

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
    token_input = layers.Input(shape=input_shape, name="seman_input_token")
    segment_input = layers.Input(shape=input_shape, name="seman_input_segment")

    embedding = BertEmbedding(
        config=BertConfig(max_sequence_length=MAX_LEN, name="seman_bert")
    )([token_input, segment_input])

    # First convolutional block
    conv1 = layers.Conv1D(64, 5, activation="relu", name="seman_conv1")(embedding)
    pool1 = layers.MaxPooling1D(3, name="seman_pool1")(conv1)

    # Second convolutional block
    conv2 = layers.Conv1D(128, 5, activation="relu", name="seman_conv2")(pool1)
    pool2 = layers.MaxPooling1D(3, name="seman_pool2")(conv2)

    # Third convolutional block
    conv3 = layers.Conv1D(256, 5, activation="relu", name="seman_conv3")(pool2)

    # Bidirectional LSTM
    gru = layers.Bidirectional(
        layers.LSTM(64, name="seman_lstm", return_sequences=False), name="seman_gru"
    )(conv3)

    return token_input, segment_input, gru


def create_semantic_model(learning_rate: float = DEFAULT_LEARNING_RATE) -> keras.Model:
    """
    Create the semantic model for the bert encoding.
    :param learning_rate: The learning rate of the model.
    :return: The model.
    """
    token_input, segment_input, gru = create_semantic_extractor()
    classification_output = create_classification_layers(gru)

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
    model_input = layers.Input(shape=input_shape, name="vis_input")

    # First convolutional block
    conv1 = layers.Conv2D(
        filters=64, kernel_size=3, padding="same", activation="relu", name="vis_conv1"
    )(model_input)
    pool1 = layers.MaxPooling2D(pool_size=2, strides=2, name="vis_pool1")(conv1)

    # Second convolutional block
    conv2 = layers.Conv2D(
        filters=128, kernel_size=3, padding="same", activation="relu", name="vis_conv2"
    )(pool1)
    pool2 = layers.MaxPooling2D(pool_size=2, strides=2, name="vis_pool2")(conv2)

    # Third convolutional block
    conv3 = layers.Conv2D(
        filters=256, kernel_size=3, padding="same", activation="relu", name="vis_conv3"
    )(pool2)
    pool3 = layers.MaxPooling2D(pool_size=2, strides=2, name="vis_pool3")(conv3)

    # Fourth convolutional block
    conv4 = layers.Conv2D(
        filters=512, kernel_size=3, padding="same", activation="relu", name="vis_conv4"
    )(pool3)
    pool4 = layers.MaxPooling2D(pool_size=2, strides=2, name="vis_pool4")(conv4)

    # Flatten layer
    flattened = layers.Flatten(name="vis_flatten")(pool4)

    return model_input, flattened


def create_visual_model(learning_rate: float = DEFAULT_LEARNING_RATE) -> keras.Model:
    """
    Create the visual model for the image encoding.
    :param learning_rate: The learning rate of the model.
    :return: The model.
    """
    image_input, image_flatten = create_visual_extractor()
    classification_output = create_classification_layers(image_flatten)

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

    classification_output = create_classification_layers(concatenated)

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
        self.name = kwargs.pop("name", "BertEmbedding")
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

    config = None

    def __init__(self, config):
        super().__init__(name=config.name)
        self.config = config

        self.token_embedding = self.add_weight(
            shape=[self.config.vocab_size, self.config.hidden_size],
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
        )
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

    def build(self, input_shape):
        """
        Build the layer.
        """
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
        return self.dropout(embeddings, training=training)

    def get_config(self):
        """
        Get the configuration of the layer.
        :return: The configuration.
        """
        config = super().get_config()
        config.update(self.config.__dict__)
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create a layer from the configuration.
        :param config: The configuration.
        :return: The layer.
        """
        return cls(BertConfig(**config))
