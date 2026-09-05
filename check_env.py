import torch

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("compiled arch list:", torch.cuda.get_arch_list())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("total VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))