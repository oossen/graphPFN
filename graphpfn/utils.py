from typing import Callable, Type
from pfns.bar_distribution import FullSupportBarDistribution
from pfns.bar_distribution import get_bucket_limits
from nanotabpfn.utils import get_default_device

import torch
from torch.utils.data import DataLoader


def make_bar_distribution(prior_factory: Callable[[int], DataLoader],
                          n_buckets: int = 100,
                          n_samples: int = 10000
                          ) -> FullSupportBarDistribution:
    """
    Construct a full support bar/Riemann distribution.
    Sample `n_samples` many data tables from `prior` and
    choose `n_buckets` many buckets to each contain the same number of targets.
    
    The argument `prior_factory` takes one int argument (number of samples) and returns a dataloader.
    """
    sampled_ys = []
    prior = iter(prior_factory(n_samples))
    for data in prior:
        y = data['y']
        for col in y: # only one iteration if batch_size==1
            sampled_ys.append(col.unsqueeze(-1))
    ys_tensor = torch.concat(sampled_ys)
    
    device = get_default_device()
    buckets = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    print(f"Buckets of bar distribution: {buckets}")
    return FullSupportBarDistribution(buckets)