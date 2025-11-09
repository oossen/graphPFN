from functools import partial
from typing import List
from datetime import datetime

import torch

from graphpfn.callbacks import SanityCheckLinearLoggerCallback, SanityCheckLoggerCallback
from graphpfn.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.callbacks import Callback, TensorboardLoggerCallback

from graphpfn.utils import make_bar_distribution
from priors.linear_dataloader import LinearDataLoader


from configs.linear_configs import training_config as args


device = get_default_device()

prior = LinearDataLoader(num_steps=args["steps"], batch_size=args["batchsize"], seed=42)

model = args["model"]
n_buckets = model.num_outputs

prior_factory = partial(LinearDataLoader, batch_size=10, seed=42)
dist, buckets = make_bar_distribution(prior_factory, n_buckets=n_buckets, n_samples=args["n_bardist_samples"])

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"{args['output']}/{datetime_str}"
tensorboard_dir = f"{output_dir}/tensorboard"
test_prior_factory = partial(LinearDataLoader, batch_size=1, seed=42)
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
