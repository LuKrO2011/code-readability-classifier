import torch

print("CUDA is available for torch: ", torch.cuda.is_available())
print("PyTorch cuDNN version:", torch.backends.cudnn.version())  # 8700
print("Cuda version:", torch.version.cuda)  # 11.8
