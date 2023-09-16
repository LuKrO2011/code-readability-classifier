import tensorflow as tf

# Check if GPU is available
if tf.config.list_physical_devices("GPU"):
    print("GPU is available")
    print("CuDNN is enabled: True")
else:
    print("GPU/CuDNN is not available for tensorflow/keras"),

import torch

print("CUDA is available for torch: ", torch.cuda.is_available())
