from graphpfn.baseline_model import BaselineModel
from pfns.bar_distribution import get_bucket_limits

import torch
import torch.nn as nn


# activation functions
class ArcsinhWrapper(nn.Module):
    def __init__(self, activation: nn.Module, swap_sign=False):
        super().__init__()
        self.activation = activation
        self.swap_sign = swap_sign
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        if self.swap_sign:
            x = -x
        return torch.asinh(x)
    
class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)
    

num_outputs = 1000
low, high = -5.0, 5.0
buckets = get_bucket_limits(num_outputs=num_outputs, full_range=(low, high))
bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0


prior_config = {
    
    "dataset_config": {
        # number of train samples per dataset
        # int
        "number_train_samples_per_dataset": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 5, "high": 100}
        },
        # number of test samples per dataset
        # can be fixed because architecture is agnostic to the number of test samples
        # int
        "number_test_samples_per_dataset": {  # number of test samples per dataset. Can be fixed because architecture is agnostic to the number of test samples.
            "value": 20
        },
    },

    "graph_config": {
        # number of nodes in the causal graph
        # each node may contain several features
        # one of these will become the target, the others (if not dropped) features of the generated data
        # int
        "num_nodes": { 
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 5, "high": 25}
        },
        # probability that any two nodes in the causal graph are connected
        # float
        "edge_prob": {
            "distribution": "logarithmic",
            "distribution_parameters": {"low": 0.1, "high": 0.4}
        },
    },

    "noise_config": {    
        # the standard deviation of noise sampled at root nodes when propagating through the SCM
        # float
        "root_std_dist": {
            "distribution": "shifted_exponential",
            "distribution_parameters": {"rate": 1 / 1.0, "shift": 0.1}
        },
        # the standard deviation of noise sampled at non-root nodes when propagating through the SCM
        # float
        "non_root_std_dist": {
            "distribution": "shifted_exponential",
            "distribution_parameters": {"rate": 1 / 0.1, "shift": 0.1}
        },
    },
    
    "activations": [ArcsinhWrapper(nn.Identity()), 
                    ArcsinhWrapper(nn.LeakyReLU(negative_slope=0.1)), 
                    ArcsinhWrapper(Square()),
                    ArcsinhWrapper(nn.Identity(), swap_sign=True), 
                    ArcsinhWrapper(nn.LeakyReLU(negative_slope=0.1), swap_sign=True), 
                    ArcsinhWrapper(Square(), swap_sign=True)],
    
    "bucket_mids": bucket_mids,
}


training_config = {
    # path to save the trained model to
    # str
    "saveweights": "baseline",
    
    # the model we are training
    # BaselineModel
    "model": BaselineModel(
        num_attention_heads=8,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=num_outputs,
    ),
    
    # batch size used during training
    # int
    "batchsize": 2,
    
    # learning rate
    # float
    "lr": 1e-4,
    
    # number of data batches contained in each epoch
    # int
    "steps": 5000,
    
    # number of epochs to train for
    # int
    "epochs": 100,
    
    # whether to train with NLL
    # bool
    "nll": False,
    
    # the buckets used for the bar distribution
    # torch.Tensor
    "buckets": buckets,
}