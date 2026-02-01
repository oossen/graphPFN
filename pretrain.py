from functools import partial
from typing import List
from datetime import datetime

from graphpfn.callbacks import SanityCheckLoggerCallback
from graphpfn.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.callbacks import Callback, TensorboardLoggerCallback

from visualization.make_visualization import plot_all

from priors.observational_dataloader import ObservationalDataLoader
from configs.default_configs import prior_config, training_config as args


device = get_default_device()

prior = ObservationalDataLoader(num_steps=args["steps"],
                                batch_size=args["batchsize"],
                                prior_config=prior_config,
                                seed=42)

model = args["model"]

now = datetime.now()
datetime_str = now.strftime("%m_%d_%H_%M")
run_name = f"{args['saveweights']}_{datetime_str}"
output_dir = f"workdir/{run_name}"
tensorboard_dir = f"{output_dir}/tensorboard"
test_prior_factory = partial(ObservationalDataLoader, batch_size=1, prior_config=prior.prior_config, seed=43)
sanity_callback = SanityCheckLoggerCallback(tensorboard_dir, test_prior_factory)
logger_callback = TensorboardLoggerCallback(tensorboard_dir)
callbacks: List[Callback] = [logger_callback, sanity_callback]

# visualize data and save configs
visualization_prior = ObservationalDataLoader(10, 1, prior_config=prior.prior_config, seed=44)
plot_all(visualization_prior, f"{output_dir}/visualization")
    
trained_model, loss = train(
    model=model,
    prior=prior,
    buckets=args["buckets"].to(device),
    epochs=args["epochs"],
    lr=args["lr"],
    callbacks=callbacks,
    run_name=run_name,
)
