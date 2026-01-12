from tfmplayground.model import NanoTabPFNModel


prior_config = {
    "n_train_samples": 5,
    "n_test_samples": 1,
    "fixed_graph_ratio": 0.0,
}


training_config = {
    # path to save the trained model to
    # str
    "saveweights": "ppd",
    
    # the model we are training
    # NanoTabPFNModel
    "model": NanoTabPFNModel(
        num_attention_heads=8,
        embedding_size=192,
        mlp_hidden_size=768,
        num_layers=6,
        num_outputs=500,
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
    "epochs": 60,
}