from functools import partial
from typing import List
from datetime import datetime
import networkx as nx

import torch

from graphpfn.callbacks import SanityCheckLinearLoggerCallback, SanityCheckLoggerCallback
from graphpfn.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.callbacks import Callback, TensorboardLoggerCallback

from graphpfn.utils import make_bar_distribution
from priors.basic_dataloader import BasicDataLoader


from configs.default_configs import training_config as args

g_0 = nx.DiGraph()
g_0.add_edges_from([(0, 1), (1, 2)])
g_1 = nx.DiGraph()
g_1.add_edges_from([(1, 0), (0, 2)])
g_2 = nx.DiGraph()
g_2.add_edges_from([(0, 2), (2, 1)])
graphs = [g_0, g_1, g_2]
graph_subset = [g_0]


device = get_default_device()

prior = BasicDataLoader(num_steps=args["steps"], batch_size=args["batchsize"], graphs=graph_subset, seed=42)

model = args["model"]
n_buckets = model.num_outputs

prior_factory = partial(BasicDataLoader, batch_size=10, graphs=graph_subset, seed=42)
dist, buckets = make_bar_distribution(prior_factory, n_buckets=n_buckets, n_samples=args["n_bardist_samples"])

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"{args['output']}/{datetime_str}"
tensorboard_dir = f"{output_dir}/tensorboard"
test_prior_factory = partial(BasicDataLoader, batch_size=1, graphs=graphs, seed=42)
sanity_callback = SanityCheckLoggerCallback(tensorboard_dir, test_prior_factory)
sanity_callback_linear = SanityCheckLinearLoggerCallback(tensorboard_dir, test_prior_factory)
logger_callback = TensorboardLoggerCallback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, sanity_callback, sanity_callback_linear]

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


model_params = {'architecture': {
                    'num_layers': int(model.num_layers),
                    'embedding_size': int(model.embedding_size),
                    'num_attention_heads': int(model.num_attention_heads),
                    'num_graph_attention_heads': model.num_graph_attention_heads if hasattr(model, 'num_graph_attention_heads') else None,
                    'gcn_hidden_size': model.gcn_hidden_size if hasattr(model, 'gcn_hidden_size') else None,
                    'mlp_hidden_size': int(model.mlp_hidden_size),
                    'num_outputs': int(model.num_outputs)
                },
                'model': model.state_dict(),}
torch.save(model_params, f"{args['saveweights']}_model.pth")
torch.save(buckets.to('cpu'), f"{args['saveweights']}_dist.pth")
