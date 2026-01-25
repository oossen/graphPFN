from tfmplayground.model import NanoTabPFNModel
from pfns.bar_distribution import get_bucket_limits
from tfmplayground.utils import get_default_device


n_outputs = 200
low, high = -5.0, 5.0
device = get_default_device()
buckets = get_bucket_limits(num_outputs=n_outputs, full_range=(low, high)).to(device)
bucket_mids = (buckets[:-1] + buckets[1:]) / 2.0


prior_config = {
    
    "dataset_config": {
        # number of train samples per dataset
        # int
        "number_train_samples_per_dataset": {
            "value": 5,
        },
        # number of test samples per dataset
        # can be fixed because architecture is agnostic to the number of test samples
        # int
        "number_test_samples_per_dataset": {  # number of test samples per dataset. Can be fixed because architecture is agnostic to the number of test samples.
            "value": 1
        },
    },

    "graph_config": {
        # number of nodes in the causal graph
        # each node may contain several features
        # one of these will become the target, the others (if not dropped) features of the generated data
        # int
        "num_nodes": { 
            "value": 5,
        },
        # probability that any two nodes in the causal graph are connected
        # float
        "edge_prob": {
            "distribution": "logarithmic",
            "distribution_parameters": {"low": 0.1, "high": 0.4}
        },
    },

    "scm_config": {    
        # the standard deviation of noise sampled at root nodes when propagating through the SCM
        # float
        "root_std": {
            "distribution": "logarithmic",
            "distribution_parameters": {"low": 0.1, "high": 1.0}
        },
        # the standard deviation of noise sampled at non-root nodes when propagating through the SCM
        # float
        "non_root_std": {
            "distribution": "logarithmic",
            "distribution_parameters": {"low": 0.1, "high": 0.5}
        },
    },
    
    "bucket_mids": bucket_mids,
}


training_config = {
    # path to save the trained model to
    # str
    "saveweights": "likelihood_training",
    
    # the model we are training
    # NanoTabPFNModel
    "model": NanoTabPFNModel(
        num_attention_heads=8,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=n_outputs,
    ),
    
    # the bucket limits used for defining the bar distribution
    "buckets": buckets,
    
    # batch size used during training
    # int
    "batchsize": 1,
    
    # number of gradients to accumulate before updating weights
    # int
    "accumulate": 1,
    
    # learning rate
    # float
    "lr": 1e-4,
    
    # number of data batches contained in each epoch
    # int
    "steps": 5000,
    
    # number of epochs to train for
    # int
    "epochs": 30,
}