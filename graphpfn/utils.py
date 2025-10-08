from pfns.bar_distribution import FullSupportBarDistribution
from pfns.bar_distribution import get_bucket_limits
from nanotabpfn.utils import get_default_device

import torch
from torch.utils.data import DataLoader

def make_bar_distribution(prior: DataLoader,
                          n_buckets: int = 100,
                          n_samples: int = 10000
                          ) -> FullSupportBarDistribution:
    """
    Construct a full support bar/Riemann distribution.
    Sample `n_samples` many data tables from `prior` and
    choose `n_buckets` many buckets to each contain the same number of targets.
    """
    sampled_ys = []
    n_sampled = 0
    prior_iter = iter(prior)
    while n_sampled < n_samples:
        try:
            data = next(prior_iter)
        except StopIteration:
            raise IndexError("Not enough data in this dataloader!")
        y = data['y']
        b, n, _ = y.shape
        n_sampled += b * n
        for col in y:
            sampled_ys.append(col.unsqueeze(-1))
    ys_tensor = torch.concat(sampled_ys)
    
    device = get_default_device()
    buckets = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    print(f"Buckets of bar distribution: {buckets}")
    return FullSupportBarDistribution(buckets)