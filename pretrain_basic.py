from functools import partial
from typing import List
from datetime import datetime

import torch

from graphpfn.callbacks import SanityCheckLoggerCallback, OldSanityCheckLoggerCallback
from graphpfn.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.callbacks import Callback, TensorboardLoggerCallback

from graphpfn.utils import make_bar_distribution
from visualization.make_visualization import plot_all

from priors.basic_dataloader import ObservationalDataLoader
from configs.ppd_configs_graph import training_config as args
fixed_graph = False


device = get_default_device()

prior = ObservationalDataLoader(num_steps=args["steps"],
                                batch_size=args["batchsize"],
                                fixed_graph=fixed_graph,
                                seed=42)

model = args["model"]
n_buckets = model.num_outputs

prior_factory = partial(ObservationalDataLoader,
                        batch_size=10,
                        fixed_graph=fixed_graph,
                        seed=42)
dist, buckets = make_bar_distribution(prior_factory, n_buckets=n_buckets, n_samples=args["n_bardist_samples"])

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
run_name = f"{args['saveweights']}_{datetime_str}"
output_dir = f"workdir/{run_name}"
tensorboard_dir = f"{output_dir}/tensorboard"
test_prior_factory = partial(ObservationalDataLoader, batch_size=1, fixed_graph=fixed_graph, seed=43)
sanity_callback = SanityCheckLoggerCallback(tensorboard_dir, test_prior_factory)
old_sanity_callback = OldSanityCheckLoggerCallback(tensorboard_dir, test_prior_factory)
logger_callback = TensorboardLoggerCallback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, sanity_callback, old_sanity_callback]

visualization_prior = ObservationalDataLoader(10, 1, fixed_graph=fixed_graph, seed=42)
plot_all(visualization_prior, f"{output_dir}/visualization")
# save buckets
with open(f"{output_dir}/buckets.txt", "w") as f:
    f.write(str(buckets))
    
torch.save(buckets.to('cpu'), f"{output_dir}/dist.pth")
trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=dist,
    epochs=args["epochs"],
    accumulate_gradients=args["accumulate"],
    lr=args["lr"],
    device=torch.device(device),
    callbacks=callbacks,
    run_name=run_name,
)
