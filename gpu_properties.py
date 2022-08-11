import torch

print("GPU Configuration\n",torch.cuda.get_device_properties(0))
