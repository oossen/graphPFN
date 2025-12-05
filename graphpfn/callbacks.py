from graphpfn.interface import Regressor

from tfmplayground.callbacks import TensorboardLoggerCallback
from tfmplayground.utils import get_default_device

from graphpfn.interface import cross_validate
from torch.utils.tensorboard.writer import SummaryWriter


class SanityCheckLoggerCallback(TensorboardLoggerCallback):
    """
    On epoch end, evaluate the model on data from the same prior that it is being trained on.
    To initialize, needs the bar distribution and prior used for training.
    """
    def __init__(self, log_dir: str, prior_factory, num_steps=2500):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.prior_factory = prior_factory
        self.num_steps = num_steps
    
    def on_epoch_end(self, epoch: int, epoch_time: float, loss: float, model, **kwargs):
        test_prior = self.prior_factory(self.num_steps)
        dist = kwargs['dist']
        regressor = Regressor(model, dist, get_default_device())
        scores = []
        for data in test_prior:
            X = data['x'][0].cpu().numpy()
            y = data['y'][0].cpu().numpy()
            single_eval_pos = data['single_eval_pos']
            scores.append(cross_validate(regressor, X, y, single_eval_pos, 5, **data['graph_information']))
        avg_score = sum(scores) / len(scores)
        self.writer.add_scalar('synthetic R²', avg_score, epoch)