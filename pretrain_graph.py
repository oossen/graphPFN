from functools import partial
from typing import List
from datetime import datetime

import torch

from nanotabpfn.evaluation import TOY_TASKS_REGRESSION
from graphpfn.callbacks import EvaluationLoggerCallback, SanityCheckLoggerCallback, SanityCheckLoggerGraphCallback
from graphpfn.train import train
from nanotabpfn.utils import get_default_device
from nanotabpfn.callbacks import Callback, TensorboardLoggerCallback

from graphpfn.model import GraphPFNModel
from graphpfn.utils import make_bar_distribution
from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from configs.default_configs import prior_config, training_config as args
from visualization.make_visualization import make_all


device = get_default_device()

prior = ObservationalDataLoader(num_steps=args["steps"],
                                batch_size=args["batchsize"],
                                prior_config=prior_config,
                                seed=42)

model = GraphPFNModel(
    num_attention_heads=args["heads"] // 2,
    num_graph_attention_heads=args["heads"] // 4,
    embedding_size=args["embeddingsize"],
    mlp_hidden_size=args["hiddensize"],
    num_layers=args["layers"],
    num_outputs=args["n_buckets"],
)

prior_factory = partial(ObservationalDataLoader,
                        batch_size=10,
                        prior_config=prior_config,
                        seed=42)
dist, buckets = make_bar_distribution(prior_factory, n_buckets=args["n_buckets"], n_samples=args["n_bardist_samples"])

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"{args['output']}/{datetime_str}"
tensorboard_dir = f"{output_dir}/tensorboard"
sanity_callback = SanityCheckLoggerGraphCallback(tensorboard_dir, prior)
logger_callback = TensorboardLoggerCallback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, sanity_callback]

# visualize data and save configs
make_all(ObservationalDataLoader, prior_config, f"{output_dir}/visualization")
# save buckets
with open(f"{output_dir}/buckets.txt", "w") as f:
    f.write(str(buckets))
    

trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args["epochs"],
    accumulate_gradients=args["accumulate"],
    lr=args["lr"],
    device=torch.device(device),
    callbacks=callbacks,
)

torch.save(trained_model.to('cpu').state_dict(), args["saveweights"])
