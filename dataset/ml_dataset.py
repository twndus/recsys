import torch
from torch.utils.data import Dataset

class ML1MDataset(Dataset):
    def __init__(self, cfg: dict):
        pass

    def split_and_get_data(self):
        train_data = None
        valid_data = None
        test_data = None
        return train_data, valid_data, test_data
