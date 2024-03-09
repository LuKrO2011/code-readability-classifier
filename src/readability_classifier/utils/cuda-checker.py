print("Torch:")
try:
    import torch

    print("PyTorch version:", torch.__version__)  # 1.9.0+cu111
    print("CUDA is available for torch: ", torch.cuda.is_available())
    print("PyTorch cuDNN version:", torch.backends.cudnn.version())  # 8700
    print("Cuda version:", torch.version.cuda)  # 11.8
except Exception as e:
    print(e)
print()

print("Tensorflow:")
try:
    import tensorflow as tf

    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    print("Is built with CUDA:", tf.test.is_built_with_cuda())
    print("Tensorflow version:", tf.__version__)
except Exception as e:
    print(e)
print()

print("Tensorflow GPU test:")
try:
    with tf.device("/device:GPU:0"):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name="a")
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name="b")
        c = tf.matmul(a, b)
        print(c)
except Exception as e:
    print(e)
print()

print("Keras GPU test:")
try:
    import numpy as np
    from keras import layers, models

    model = models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(32,)))
    model.add(layers.Dense(10, activation="softmax"))
    model.summary()
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    model.fit(data, labels, epochs=10, batch_size=32)
except Exception as e:
    print(e)
print()
