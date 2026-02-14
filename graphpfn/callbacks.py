from graphpfn.interface import Regressor
from priors.observational_dataloader import ObservationalDataLoader

from tfmplayground.callbacks import TensorboardLoggerCallback

from graphpfn.interface import cross_validate
from torch.utils.tensorboard.writer import SummaryWriter


class ValidationCallback(TensorboardLoggerCallback):
    """
    On epoch end, evaluate the model on data from the same prior that it is being trained on.
    To initialize, needs the bar distribution and prior used for training.
    """
    def __init__(self, log_dir: str, prior_config, num_steps=500, seed=100):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.prior_config = prior_config
        self.num_steps = num_steps
        self.seed = seed
    
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        test_prior = ObservationalDataLoader(self.num_steps, 1, self.prior_config, self.seed)
        regressor = Regressor(model, kwargs['buckets'])
        scores = []
        for data in test_prior:
            X = data['x'][0].cpu().numpy()
            y = data['y'][0].cpu().numpy()
            single_eval_pos = data['single_eval_pos']
            scores.append(cross_validate(regressor, X, y, single_eval_pos, 5, **data['graph_information']))
        avg_score = sum(scores) / len(scores)
        self.writer.add_scalar('synthetic R²', avg_score, epoch)