import keras
import tensorflow as tf


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
            "weight",
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
