import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_device(device)
print(f"Using device: {device}")