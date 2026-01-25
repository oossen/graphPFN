from functools import partial
from typing import List
from datetime import datetime

import torch

from graphpfn.callbacks import SanityCheckLoggerCallback, OldSanityCheckLoggerCallback
from graphpfn.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.callbacks import Callback, TensorboardLoggerCallback
from pfns.bar_distribution import FullSupportBarDistribution

from visualization.make_visualization import plot_all

from priors.basic_dataloader_graph_prior import ObservationalDataLoader
from configs.likelihood_training_configs import training_config as args, prior_config


device = get_default_device()

prior = ObservationalDataLoader(num_steps=args["steps"],
                                batch_size=args["batchsize"],
                                prior_config=prior_config,
                                seed=42)


buckets = args["buckets"].to(device)
dist = FullSupportBarDistribution(buckets)

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
run_name = f"{args['saveweights']}_{datetime_str}"
output_dir = f"workdir/{run_name}"
tensorboard_dir = f"{output_dir}/tensorboard"
test_prior_factory = partial(ObservationalDataLoader, batch_size=1, prior_config=prior_config, seed=43)
sanity_callback = SanityCheckLoggerCallback(tensorboard_dir, test_prior_factory)
old_sanity_callback = OldSanityCheckLoggerCallback(tensorboard_dir, test_prior_factory)
logger_callback = TensorboardLoggerCallback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, sanity_callback, old_sanity_callback]

visualization_prior = ObservationalDataLoader(10, 1, prior_config=prior_config, seed=44)
plot_all(visualization_prior, f"{output_dir}/visualization")
# save buckets
with open(f"{output_dir}/buckets.txt", "w") as f:
    f.write(str(buckets))
    
torch.save(buckets.to('cpu'), f"{output_dir}/dist.pth")
trained_model, loss = train(
    model=args['model'],
    prior=prior,
    criterion=dist,
    epochs=args["epochs"],
    accumulate_gradients=args["accumulate"],
    lr=args["lr"],
    device=torch.device(device),
    callbacks=callbacks,
    run_name=run_name,
)
