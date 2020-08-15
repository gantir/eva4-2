import torch


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def init_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
