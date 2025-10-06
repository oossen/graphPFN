from torch.utils.data import DataLoader
import torch


def compare_dataloaders(loader_1: DataLoader, loader_2: DataLoader):
    iter_1, iter_2 = iter(loader_1), iter(loader_2)
    for i, (data_1, data_2) in enumerate(zip(iter_1, iter_2)):
        x_1 = data_1['x']
        y_1 = data_1['y']
        pos_1 = data_1['single_eval_pos']
        x_2 = data_2['x']
        y_2 = data_2['y']
        pos_2 = data_2['single_eval_pos']
        
        if not torch.allclose(x_1, x_2):
            raise ValueError(f"In databatch {i}, the features are not the same.")
        if not torch.allclose(y_1, y_2):
            raise ValueError(f"In databatch {i}, the targets are not the same.")
        if not pos_1 == pos_2:
            raise ValueError(f"In databatch {i}, the train test splits are not the same.")
        
    if next(iter_1, None) is not None or next(iter_2, None) is not None:
        raise ValueError(f"The dataloaders do not have the same number of data batches.")
        
    print("All good!")