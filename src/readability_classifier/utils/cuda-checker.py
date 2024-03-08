import tensorflow as tf
import torch

print("Tensorflow:")
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
print("Tensorflow version:", tf.__version__)
print()

print("Torch:")
print("PyTorch version:", torch.__version__)  # 1.9.0+cu111
print("CUDA is available for torch: ", torch.cuda.is_available())
print("PyTorch cuDNN version:", torch.backends.cudnn.version())  # 8700
print("Cuda version:", torch.version.cuda)  # 11.8
print()
