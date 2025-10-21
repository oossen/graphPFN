from typing import Any, Dict, Iterator

import torch
from prior.dataloaders.observational_dataloader import ObservationalDataLoader


class ConstantDataLoader(ObservationalDataLoader):
    """Like `ObservationalDataLoader`, but yielding the same iterator every time."""
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)
        return iter(self.batch_function() for _ in range(self.num_steps))