import torch

g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Default Torch device: {g_device}")
