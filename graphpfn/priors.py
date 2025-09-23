from typing import Callable, Iterator
import torch
from torch.utils.data import DataLoader

class PriorDataLoader(DataLoader):
    def __init__(self, get_batch_function: Callable, num_steps: int, batch_size: int, num_datapoints_max: int, num_features: int, device: torch.device):
        self.get_batch_function = get_batch_function
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device

    def __iter__(self) -> Iterator:
        return iter(self.get_batch_function(self.batch_size, self.num_datapoints_max, self.num_features) for _ in range(self.num_steps))

    def __len__(self) -> int:
        return self.num_steps