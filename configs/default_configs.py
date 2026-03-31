from graphpfn.baseline_model import BaselineModel
from pfns.bar_distribution import get_bucket_limits

from configs.tabicl_activations import get_activations
    

num_outputs = 1000
low, high = -5.0, 5.0
buckets = get_bucket_limits(num_outputs=num_outputs, full_range=(low, high))


prior_config = {
    
    "dataset_config": {
        # number of train samples per dataset
        # int
        "number_train_samples_per_dataset": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 2, "high": 256}
        },
        # number of test samples per dataset
        # can be fixed because architecture is agnostic to the number of test samples
        # int
        "number_test_samples_per_dataset": {  # number of test samples per dataset. Can be fixed because architecture is agnostic to the number of test samples.
            "value": 100
        },
    },

    "graph_config": {
        # number of nodes in the causal graph
        # each node may contain several features
        # one of these will become the target, the others (if not dropped) features of the generated data
        # int
        "num_nodes": { 
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 3, "high": 30}
        },
        # probability that any two nodes in the causal graph are connected
        # float
        "edge_prob": {
            "distribution": "logarithmic",
            "distribution_parameters": {"low": 0.05, "high": 0.5}
        },
        # the number of features contained in each node
        # int
        "features_per_node": {
            "value": 1,
        },
        # the mode for creating the probabilistic adjacency matrix
        # categorical over "binary", "beta", "uncertain"
        "prob_adj_mode": {
            "distribution": "categorical",
            "distribution_parameters": {"choices": ["binary", "beta", "uncertain"], "probabilities": [0.0, 1.0, 0.0]}
        },
    },

    "scm_config": {    
        # the standard deviation of noise sampled at root nodes when propagating through the SCM
        # float
        "root_std_dist": {
            "distribution": "shifted_exponential",
            "distribution_parameters": {"rate": 1 / 1.0, "shift": 0.2}
        },
        # the standard deviation of noise sampled at non-root nodes when propagating through the SCM
        # float
        "non_root_std_dist": {
            "distribution": "shifted_exponential",
            "distribution_parameters": {"rate": 1 / 0.1, "shift": 0.1}
        },
        # the activation functions to be used in the SCM
        # categorical distribution over nn.Modules
        "activation_dist": {
            "distribution": "categorical",
            "distribution_parameters": {"choices": get_activations()}
        },
        # the probability that a given feature becomes categorical
        # float
        "categorical_prob": {
            "distribution": "uniform",
            "distribution_parameters": {"low": 0.0, "high": 0.3},
        },
        # the number of categories to use for a given categorical feature
        # int
        "num_categories": {
            "distribution": "discrete_uniform",
            "distribution_parameters": {"low": 2, "high": 10},
        },
    },
    
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
    "batchsize": 4,
    
    # learning rate
    # float
    "lr": 1e-4,
    
    # number of steps to accumulate gradients for
    # int
    "accumulate_gradients": 1,
    
    # number of data batches contained in each epoch
    # int
    "steps": 10000,
    
    # number of epochs to train for
    # int
    "epochs": 300,
    
    # the buckets used for the bar distribution
    # torch.Tensor
    "buckets": buckets,
}