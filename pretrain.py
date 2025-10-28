from functools import partial
from typing import List
from datetime import datetime

import torch

from nanotabpfn.evaluation import TOY_TASKS_REGRESSION
from graphpfn.callbacks import EvaluationLoggerCallback, SanityCheckLoggerCallback
from graphpfn.train import train
from nanotabpfn.utils import get_default_device
from nanotabpfn.callbacks import Callback, TensorboardLoggerCallback

from graphpfn.utils import make_bar_distribution
from prior.dataloaders.observational_dataloader import ObservationalDataLoader
from visualization.make_visualization import make_all


from configs.default_configs import prior_config, training_config as args


device = get_default_device()

prior = ObservationalDataLoader(num_steps=args["steps"],
                                batch_size=args["batchsize"],
                                prior_config=prior_config,
                                seed=42)

model = args["model"]
n_buckets = model.num_outputs

prior_factory = partial(ObservationalDataLoader,
                        batch_size=10,
                        prior_config=prior_config,
                        seed=42)
dist, buckets = make_bar_distribution(prior_factory, n_buckets=n_buckets, n_samples=args["n_bardist_samples"])

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
output_dir = f"{args['output']}/{datetime_str}"
tensorboard_dir = f"{output_dir}/tensorboard"
evaluation_callback = EvaluationLoggerCallback(tensorboard_dir, TOY_TASKS_REGRESSION, prior)
sanity_callback = SanityCheckLoggerCallback(tensorboard_dir, prior)
logger_callback = TensorboardLoggerCallback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, evaluation_callback, sanity_callback]

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


model_params = {'architecture': {
                    'num_layers': int(model.num_layers),
                    'embedding_size': int(model.embedding_size),
                    'num_attention_heads': int(model.num_attention_heads),
                    'num_graph_attention_heads': model.num_graph_attention_heads if hasattr(model, 'num_graph_attention_heads') else None,
                    'mlp_hidden_size': int(model.mlp_hidden_size),
                    'num_outputs': int(model.num_outputs)
                },
                'model': model.state_dict(),}
torch.save(model_params, f"{args['saveweights']}_model.pth")
torch.save(buckets.to('cpu'), f"{args['saveweights']}_dist.pth")
