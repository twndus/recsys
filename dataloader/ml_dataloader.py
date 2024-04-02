import torch
from torch.utils.data import DataLoader

from dataset.ml_dataset import ML1MDataset

class ML1MDataLoader(DataLoader):

    def __init__(self, cfg: dict, dataset: ML1MDataset, train: bool=True):
        pass

    def __getitem__(self, index: int) -> dict:
        return {'X': None, 'y': None} 
