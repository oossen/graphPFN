from tfmplayground.model import NanoTabPFNModel


prior_config = {
    
    "dataset_config": {
        # number of train samples per dataset
        # int
        "number_train_samples_per_dataset": {
            "value": 50
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
            "value": 4
        },
        # probability that any two nodes in the causal graph are connected
        # float
        "edge_prob": {
            "distribution": "logarithmic",
            "distribution_parameters": {"low": 0.25, "high": 0.35}
        },
    },

    "scm_config": {    
        # the standard deviation of noise sampled at root nodes when propagating through the SCM
        # float
        "root_std": {
            "value": 1.0,
        },
        # the standard deviation of noise sampled at non-root nodes when propagating through the SCM
        # float
        "non_root_std": {
            "value": 0.1,
        },
    }
}


training_config = {
    # path to save the trained model to
    # str
    "saveweights": "probabilistic",
    
    # the model we are training
    # NanoTabPFNModel
    "model": NanoTabPFNModel(
        num_attention_heads=8,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=1000,
    ),
    
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
    
    # number of data batches used to infer buckets for bar distribution
    # int
    "n_bardist_samples": 5000,
}